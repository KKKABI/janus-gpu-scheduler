"""Nsight Compute profiler — 获取 kernel 级 memory 指标。

用法：
    from Opara.ncu_profiler import profile_ncu, merge_ncu_to_nodes
    ncu_data = profile_ncu(model_class_name, graph_module, inputs)
    merge_ncu_to_nodes(graph_module.graph.nodes, ncu_data)

ncu_data 结构：{kernel_short_name: {mem_thru, dram_thru, l2_thru, comp_thru, dur_ns}}
"""

import subprocess, csv, io, os, json, tempfile
import hashlib
import re
from collections import defaultdict


LAST_NCU_REPORT = None

_NCU_METRIC_FIELDS = {
    'Memory Throughput': 'mem_thru',
    'DRAM Throughput': 'dram_thru',
    'L2 Cache Throughput': 'l2_thru',
    'Compute (SM) Throughput': 'comp_thru',
    'Duration': 'dur_ns',
}


def _parse_ncu_number(value):
    if value is None:
        return 0.0
    value = str(value).strip().replace(',', '')
    return float(value) if value else 0.0


def parse_ncu_csv(csv_text):
    """Parse legacy NCU CSV for compatibility tests and smoke utilities."""
    lines = [line for line in csv_text.splitlines() if line.strip()]
    header_idx = next(
        (index for index, line in enumerate(lines)
         if '"Kernel Name"' in line and '"Metric Name"' in line),
        None,
    )
    if header_idx is None:
        return {}
    reader = csv.DictReader(io.StringIO('\n'.join(lines[header_idx:])))
    aggregates = {}
    for row in reader:
        kernel_name = (row.get('Kernel Name') or '').strip()
        field = _NCU_METRIC_FIELDS.get((row.get('Metric Name') or '').strip())
        if not kernel_name or not field:
            continue
        value = _parse_ncu_number(
            row.get('Average')
            if row.get('Average') not in (None, '')
            else row.get('Metric Value')
        )
        if field == 'dur_ns':
            value *= {
                'ns': 1.0, 'us': 1_000.0, 'ms': 1_000_000.0,
                's': 1_000_000_000.0,
            }.get((row.get('Metric Unit') or 'ns').strip(), 1.0)
        invocations = max(1.0, _parse_ncu_number(row.get('Invocations') or 1))
        entry = aggregates.setdefault(kernel_name, {})
        weighted_sum, weight = entry.get(field, (0.0, 0.0))
        entry[field] = (
            weighted_sum + value * invocations,
            weight + invocations,
        )
    return {
        kernel_name: {
            field: (
                fields[field][0] / fields[field][1]
                if field in fields and fields[field][1] else 0.0
            )
            for field in _NCU_METRIC_FIELDS.values()
        }
        for kernel_name, fields in aggregates.items()
    }


def profile_ncu(graph_module, inputs, ncu_bin="/usr/local/cuda-12.5/bin/ncu"):
    """用 ncu 对模型做单次推理 profiling，返回 per-kernel 指标字典。

    返回：{kernel_short_name: {mem_thru, dram_thru, l2_thru, comp_thru, dur_ns}}
    """
    import torch
    import torch._dynamo as dynamo

    # 构造独立脚本：捕获 graph module 的 forward 并用 ncu profile
    # 为了避免 ncu profile 整个框架开销，直接用 graph_module 做串行推理
    script = f'''
import torch, sys, json
torch.cuda.set_device(0)

# 重建 graph module
import torch._dynamo as dynamo
from torchvision.models import googlenet
model = googlenet(weights=None).cuda().eval()
static_inputs = [{", ".join(f"torch.zeros_like(torch.empty({list(i.shape)}), device='cuda')" for i in inputs)}]
dynamo.reset()
with torch.no_grad():
    explanation = dynamo.explain(model)(*static_inputs)
gm = explanation.graphs[0] if hasattr(explanation, 'graphs') else explanation[0]
gm.cuda()

# warmup
with torch.no_grad():
    for _ in range(3):
        gm(*static_inputs)
torch.cuda.synchronize()

# profiling pass — ncu 会捕获这一轮
with torch.no_grad():
    gm(*static_inputs)
torch.cuda.synchronize()
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    cmd = [ncu_bin, '--csv', '--print-summary', 'per-kernel',
           '--launch-count', '500', '--set', 'full',
           'python', script_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        os.unlink(script_path)
    except:
        os.unlink(script_path)
        return {}

    # 解析 CSV
    lines = [l for l in result.stdout.split('\n') if l.strip() and '==PROF' not in l]
    for i, l in enumerate(lines):
        if '"Process ID"' in l:
            header_idx = i
            break
    else:
        return {}

    csv_data = '\n'.join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_data))

    kernel_data = {}
    for row in reader:
        kname = row['Kernel Name'].split('(')[0].strip()[:55]
        if kname not in kernel_data:
            kernel_data[kname] = {}

        section = row['Section Name']
        metric = row['Metric Name']
        avg = row['Average']

        if section == 'GPU Speed Of Light Throughput':
            kernel_data[kname][metric] = float(avg) if avg else 0.0

    # 精简为需要的指标
    result = {}
    for kname, metrics in kernel_data.items():
        result[kname] = {
            'mem_thru': metrics.get('Memory Throughput', 0.0),
            'dram_thru': metrics.get('DRAM Throughput', 0.0),
            'l2_thru': metrics.get('L2 Cache Throughput', 0.0),
            'comp_thru': metrics.get('Compute (SM) Throughput', 0.0),
            'dur_ns': metrics.get('Duration', 0.0),
        }
    return result


def merge_ncu_to_nodes(nodes, ncu_data):
    """将 ncu per-kernel 数据合并到 FX 节点的 node.info 中。

    匹配逻辑：提取 kernel 函数名（最后 :: 后的部分），与 ncu key 做子串匹配。
    同名函数的不同模板实例归入同一类。
    """
    def extract_func_name(kname):
        """提取 kernel 的核心函数名"""
        # 取 < 之前的部分（去掉模板参数）
        base = kname.split('<')[0].strip()
        # 取最后一个 :: 后的部分（去掉命名空间前缀）
        parts = base.split('::')
        # 取有意义的部分（跳过 unnamed/at/native 等前缀）
        meaningful = [p for p in parts if p and p not in ('void', 'at', 'native', 'unnamed', 'ops', 'cnn',
                                                           'detail', 'impl', 'epilogue', 'cudnn', 'cublasLt')]
        if meaningful:
            return meaningful[-1]
        return parts[-1] if parts else base

    # 构建 ncu 数据索引：函数名 -> ncu_metrics
    ncu_index = {}
    for ncu_key, metrics in ncu_data.items():
        func = extract_func_name(ncu_key)
        if func:
            # 取最短且最有区分度的匹配
            if func not in ncu_index or len(ncu_key) < len(ncu_index[func][0]):
                ncu_index[func] = (ncu_key, metrics)

    for node in nodes:
        if not hasattr(node, 'info') or not node.info:
            continue

        for info_kernel in node.info:
            kname = info_kernel.get('name', '')
            if not kname:
                continue
            func = extract_func_name(kname)
            if func and func in ncu_index:
                info_kernel.update(ncu_index[func][1])

    return nodes


def _canonical_kernel_name(name):
    """Conservative full-name normalization; never collapse template types."""
    name = str(name or '').strip()
    if name.startswith('void '):
        name = name[5:]
    return re.sub(r'\s+', '', name)


def _launch_size(value):
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        result = 1
        for item in value:
            result *= int(item)
        return result
    text = str(value).strip().replace(',', '')
    numbers = [int(item) for item in re.findall(r'\d+', text)]
    if not numbers:
        return 0
    result = 1
    for item in numbers:
        result *= item
    return result


def _source_metric(entry, *names):
    metrics = entry.get('metrics', entry)
    for name in names:
        if name in metrics:
            return float(metrics[name])
    return 0.0


def _v2_source_key(entry):
    name = entry.get('name', entry.get('Name', entry.get('Kernel_Name', '')))
    grid = entry.get('grid_size', entry.get('Grid Size', entry.get('Grid', 0)))
    block = entry.get('block_size', entry.get('Block Size', entry.get('Block', 0)))
    return (_canonical_kernel_name(name), _launch_size(grid), _launch_size(block))


def _target_key(info):
    args = info.get('args', {})
    return (
        _canonical_kernel_name(info.get('name', '')),
        _launch_size(args.get('grid', 0)),
        _launch_size(args.get('block', 0)),
    )


def _function_leaf(name):
    canonical = _canonical_kernel_name(name)
    base = canonical.split('<', 1)[0]
    return base.rsplit('::', 1)[-1]


def _fallback_source_key(entry):
    exact = _v2_source_key(entry)
    return (_function_leaf(exact[0]), exact[1], exact[2])


def _fallback_target_key(info):
    exact = _target_key(info)
    return (_function_leaf(exact[0]), exact[1], exact[2])


def _profile_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_v2_identity(data, model_class_name, inputs, profile_path, graph_module):
    identity = data.get('identity', {})
    expected_shapes = [list(value.shape) for value in inputs]
    expected_dtypes = [str(value.dtype) for value in inputs]
    errors = []
    if identity.get('model_class') != model_class_name:
        errors.append('model_class')
    if identity.get('input_shapes') != expected_shapes:
        errors.append('input_shapes')
    if identity.get('input_dtypes') != expected_dtypes:
        errors.append('input_dtypes')
    try:
        import torch
        expected_device = torch.cuda.get_device_name(inputs[0].device)
    except Exception:
        expected_device = None
    if not identity.get('device_name') or identity.get('device_name') != expected_device:
        errors.append('device_name')
    try:
        expected_capability = list(torch.cuda.get_device_capability(inputs[0].device))
        expected_torch = torch.__version__
        expected_cuda = torch.version.cuda
        expected_cudnn = torch.backends.cudnn.version()
    except Exception:
        expected_capability = expected_torch = expected_cuda = expected_cudnn = None
    for field, expected in (
        ('device_capability', expected_capability),
        ('torch_version', expected_torch),
        ('cuda_version', expected_cuda),
        ('cudnn_version', expected_cudnn),
    ):
        if identity.get(field) != expected:
            errors.append(field)
    expected_sha = identity.get('profile_sha256')
    if not expected_sha or not profile_path or _profile_sha256(profile_path) != expected_sha:
        errors.append('profile_sha256')
    expected_nodes = [node.name for node in graph_module.graph.nodes]
    if identity.get('fx_node_names') != expected_nodes:
        errors.append('fx_node_names')
    expected_code_sha = hashlib.sha256(
        graph_module.code.encode('utf-8')
    ).hexdigest()
    if identity.get('fx_code_sha256') != expected_code_sha:
        errors.append('fx_code_sha256')
    return errors


def _merge_ncu_v2_by_op(nodes, sources, minimum_duration_coverage):
    source_by_op = defaultdict(list)
    for source in sources:
        source_by_op[source.get('op_name')].append(source)

    total_duration = 0.0
    mapped_duration = 0.0
    total_kernels = 0
    pending_updates = []
    mapped_operators = 0
    unmapped_records = []
    for node in nodes:
        infos = list(getattr(node, 'info', []))
        if not infos:
            continue
        node_duration = sum(max(float(info.get('dur', 0.0)), 0.0) for info in infos)
        total_duration += node_duration
        total_kernels += len(infos)
        op_sources = source_by_op.get(node.name, [])
        if not op_sources:
            unmapped_records.append((node_duration, node.name, len(infos)))
            continue
        ncu_duration = sum(max(_source_metric(source, 'dur_ns'), 1.0) for source in op_sources)
        vector = {}
        for output_name, aliases in {
            'mem_thru': ('mem_thru', 'Memory Throughput'),
            'dram_thru': ('dram_thru', 'DRAM Throughput', 'DRAM_Throughput(%)'),
            'l2_thru': ('l2_thru', 'L2 Cache Throughput'),
            'comp_thru': ('comp_thru', 'Compute (SM) Throughput', 'Compute(SM)(%)'),
        }.items():
            vector[output_name] = sum(
                max(_source_metric(source, 'dur_ns'), 1.0)
                * _source_metric(source, *aliases)
                for source in op_sources
            ) / ncu_duration
        vector['ncu_match'] = 'op_nvtx_v2'
        for info in infos:
            pending_updates.append((info, vector))
        mapped_duration += node_duration
        mapped_operators += 1

    duration_coverage = mapped_duration / total_duration if total_duration else 0.0
    accepted = duration_coverage >= minimum_duration_coverage
    if accepted:
        for info, metrics in pending_updates:
            info.update(metrics)
    examples = [
        {'op_name': name, 'duration': duration, 'kernels': kernels}
        for duration, name, kernels in sorted(
            unmapped_records, key=lambda item: item[0], reverse=True
        )[:8]
    ]
    return {
        'status': 'accepted' if accepted else 'coverage_below_threshold',
        'mapping_mode': 'op_nvtx_v2',
        'mapped_operators': mapped_operators,
        'mapped_kernels': len(pending_updates),
        'total_kernels': total_kernels,
        'unmapped_operators': len(unmapped_records),
        'unmapped_examples': examples,
        'duration_coverage': duration_coverage,
        'minimum_duration_coverage': minimum_duration_coverage,
    }


def merge_ncu_v2_to_nodes(nodes, data, minimum_duration_coverage=0.5):
    """Map ordered per-launch NCU rows by full name + grid/block + occurrence.

    A key is accepted only when NCU and Nsight contain the same number of
    launches.  Count mismatches are ambiguous and receive no metrics.
    """
    sources = list(data.get('kernels', []))
    if sources and all(source.get('op_name') for source in sources):
        return _merge_ncu_v2_by_op(nodes, sources, minimum_duration_coverage)
    targets = []
    total_duration = 0.0
    for node in nodes:
        for info in getattr(node, 'info', []):
            duration = max(float(info.get('dur', 0.0)), 0.0)
            targets.append((info, duration))
            total_duration += duration

    matched = {}
    used_sources = set()

    def match_groups(source_key, target_key, match_kind):
        source_groups = defaultdict(list)
        target_groups = defaultdict(list)
        for index, source in enumerate(sources):
            if index not in used_sources:
                source_groups[source_key(source)].append(index)
        for index, (info, _) in enumerate(targets):
            if index not in matched:
                target_groups[target_key(info)].append(index)
        for key, target_indexes in target_groups.items():
            source_indexes = source_groups.get(key, [])
            if source_indexes and len(source_indexes) == len(target_indexes):
                for target_index, source_index in zip(target_indexes, source_indexes):
                    matched[target_index] = (source_index, match_kind)
                    used_sources.add(source_index)

    match_groups(_v2_source_key, _target_key, 'exact_v2')
    match_groups(_fallback_source_key, _fallback_target_key, 'ordered_fallback_v2')

    unmatched_source_fallback = defaultdict(list)
    for index, source in enumerate(sources):
        if index not in used_sources:
            unmatched_source_fallback[_fallback_source_key(source)].append(index)

    ambiguous = 0
    unmapped = 0
    ambiguous_records = []
    unmapped_records = []
    for index, (info, _) in enumerate(targets):
        if index in matched:
            continue
        if unmatched_source_fallback.get(_fallback_target_key(info)):
            ambiguous += 1
            ambiguous_records.append((targets[index][1], info))
        else:
            unmapped += 1
            unmapped_records.append((targets[index][1], info))

    def examples(records):
        output = []
        for duration, info in sorted(records, key=lambda item: item[0], reverse=True)[:8]:
            args = info.get('args', {})
            output.append({
                'name': info.get('name', ''),
                'duration': duration,
                'grid_size': _launch_size(args.get('grid', 0)),
                'block_size': _launch_size(args.get('block', 0)),
            })
        return output

    pending_updates = []
    mapped_duration = 0.0
    exact_mapped = 0
    ordered_fallback_mapped = 0
    for target_index, (source_index, match_kind) in matched.items():
        info, duration = targets[target_index]
        source = sources[source_index]
        pending_updates.append((info, {
            'mem_thru': _source_metric(source, 'mem_thru', 'Memory Throughput'),
            'dram_thru': _source_metric(source, 'dram_thru', 'DRAM Throughput', 'DRAM_Throughput(%)'),
            'l2_thru': _source_metric(source, 'l2_thru', 'L2 Cache Throughput'),
            'comp_thru': _source_metric(source, 'comp_thru', 'Compute (SM) Throughput', 'Compute(SM)(%)'),
            'ncu_match': match_kind,
        }))
        mapped_duration += duration
        if match_kind == 'exact_v2':
            exact_mapped += 1
        else:
            ordered_fallback_mapped += 1

    duration_coverage = (
        mapped_duration / total_duration if total_duration > 0.0 else 0.0
    )
    accepted = duration_coverage >= minimum_duration_coverage
    if accepted:
        for info, metrics in pending_updates:
            info.update(metrics)

    return {
        'status': 'accepted' if accepted else 'coverage_below_threshold',
        'mapped_kernels': len(pending_updates),
        'exact_mapped_kernels': exact_mapped,
        'ordered_fallback_mapped_kernels': ordered_fallback_mapped,
        'ambiguous_kernels': ambiguous,
        'unmapped_kernels': unmapped,
        'ambiguous_examples': examples(ambiguous_records),
        'unmapped_examples': examples(unmapped_records),
        'total_kernels': len(targets),
        'duration_coverage': duration_coverage,
        'minimum_duration_coverage': minimum_duration_coverage,
    }


def profile_and_merge(graph_module, inputs, model_class_name, profile_path=None):
    """Load an identity-checked, per-launch v2 cache and merge exact matches.

    Legacy short-name dictionaries are disabled by default because they cannot
    distinguish templates, shapes, launch geometry, or occurrences.  They can
    be enabled only for smoke tests with ``JANUS_ALLOW_LEGACY_NCU=1``.
    """
    path = os.path.abspath(os.path.dirname(__file__))
    cache_dir = os.path.abspath(os.getenv(
        'JANUS_NCU_CACHE_DIR', os.path.join(path, 'ncu_result')
    ))
    cache_file = os.path.join(cache_dir, f'{model_class_name}.ncu.v2.json')
    legacy_file = os.path.join(cache_dir, f'{model_class_name}.ncu.json')
    report = {
        'model': model_class_name,
        'cache_dir': cache_dir,
        'cache_file': cache_file,
        'experimental_valid': False,
    }

    if os.path.exists(cache_file):
        report['cache_sha256'] = _profile_sha256(cache_file)
        with open(cache_file) as handle:
            data = json.load(handle)
        if data.get('aggregation') is not None:
            report['aggregation'] = data.get('aggregation')
        if data.get('schema_version') != 2:
            report['status'] = 'unsupported_schema'
        else:
            identity_errors = _validate_v2_identity(
                data, model_class_name, inputs, profile_path, graph_module
            )
            if identity_errors:
                report.update({
                    'status': 'identity_mismatch',
                    'identity_errors': identity_errors,
                })
            else:
                minimum = float(os.getenv(
                    'JANUS_NCU_MIN_DURATION_COVERAGE', '0.50'
                ))
                report.update(merge_ncu_v2_to_nodes(
                    graph_module.graph.nodes,
                    data,
                    minimum_duration_coverage=minimum,
                ))
                report['experimental_valid'] = report['status'] == 'accepted'
    elif os.path.exists(legacy_file) and os.getenv(
        'JANUS_ALLOW_LEGACY_NCU', '0'
    ) == '1':
        with open(legacy_file) as handle:
            legacy = json.load(handle)
        merge_ncu_to_nodes(graph_module.graph.nodes, legacy)
        total_duration = 0.0
        mapped_duration = 0.0
        mapped = 0
        total = 0
        for node in graph_module.graph.nodes:
            for info in getattr(node, 'info', []):
                duration = max(float(info.get('dur', 0.0)), 0.0)
                total_duration += duration
                total += 1
                if any(float(info.get(name, 0.0)) > 0.0 for name in (
                    'dram_thru', 'l2_thru', 'comp_thru'
                )):
                    mapped += 1
                    mapped_duration += duration
        report.update({
            'status': 'legacy_heuristic',
            'legacy_file': legacy_file,
            'mapped_kernels': mapped,
            'total_kernels': total,
            'duration_coverage': (
                mapped_duration / total_duration if total_duration else 0.0
            ),
        })
    else:
        report.update({
            'status': 'legacy_disabled' if os.path.exists(legacy_file) else 'cache_missing',
            'legacy_file': legacy_file if os.path.exists(legacy_file) else None,
        })

    global LAST_NCU_REPORT
    LAST_NCU_REPORT = report
    graph_module._janus_ncu_report = report
    if os.getenv('JANUS_NCU_REPORT', '0') == '1':
        print('[NCU] ' + json.dumps(report, sort_keys=True))
    return report

def has_ncu_data(nodes):
    """检查节点是否有 ncu memory 数据"""
    for node in nodes:
        if hasattr(node, 'info'):
            for info in node.info:
                if info.get('dram_thru', 0) > 0 or info.get('mem_thru', 0) > 0:
                    return True
    return False
