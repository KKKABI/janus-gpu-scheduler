"""Nsight Compute profiler — 获取 kernel 级 memory 指标。

用法：
    from Opara.ncu_profiler import profile_ncu, merge_ncu_to_nodes
    ncu_data = profile_ncu(model_class_name, graph_module, inputs)
    merge_ncu_to_nodes(graph_module.graph.nodes, ncu_data)

ncu_data 结构：{kernel_short_name: {mem_thru, dram_thru, l2_thru, comp_thru, dur_ns}}
"""

import subprocess, csv, io, os, json, tempfile


_NCU_METRIC_FIELDS = {
    'Memory Throughput': 'mem_thru',
    'DRAM Throughput': 'dram_thru',
    'L2 Cache Throughput': 'l2_thru',
    'Compute (SM) Throughput': 'comp_thru',
    'Duration': 'dur_ns',
}


def _parse_ncu_number(value):
    """Parse NCU's locale-formatted numeric cells (for example ``1,024``)."""
    if value is None:
        return 0.0
    value = str(value).strip().replace(',', '')
    return float(value) if value else 0.0


def parse_ncu_csv(csv_text):
    """Parse per-launch or per-kernel-summary NCU CSV into cache entries."""
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
                'ns': 1.0,
                'us': 1_000.0,
                'ms': 1_000_000.0,
                's': 1_000_000_000.0,
            }.get((row.get('Metric Unit') or 'ns').strip(), 1.0)
        invocations = max(1.0, _parse_ncu_number(row.get('Invocations') or 1))
        entry = aggregates.setdefault(kernel_name, {})
        weighted_sum, weight = entry.get(field, (0.0, 0.0))
        entry[field] = (
            weighted_sum + value * invocations,
            weight + invocations,
        )

    result = {}
    for kernel_name, fields in aggregates.items():
        result[kernel_name] = {
            field: (
                fields[field][0] / fields[field][1]
                if field in fields and fields[field][1] else 0.0
            )
            for field in _NCU_METRIC_FIELDS.values()
        }
    return result


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

    return parse_ncu_csv(result.stdout)


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


def profile_and_merge(graph_module, inputs, model_class_name):
    """便捷函数：从缓存加载 ncu 数据并合并到 node.info 中。

    缓存路径: Opara/ncu_result/<ModelClass>.ncu.json
    如果缓存不存在，跳过（不自动运行 ncu，需手动建缓存）。
    """
    path = os.path.abspath(os.path.dirname(__file__))
    cache_dir = os.path.join(path, 'ncu_result')
    cache_file = os.path.join(cache_dir, f'{model_class_name}.ncu.json')

    if not os.path.exists(cache_file):
        return {}

    with open(cache_file) as f:
        ncu_data = json.load(f)

    if ncu_data:
        merge_ncu_to_nodes(graph_module.graph.nodes, ncu_data)

    return ncu_data

def has_ncu_data(nodes):
    """检查节点是否有 ncu memory 数据"""
    for node in nodes:
        if hasattr(node, 'info'):
            for info in node.info:
                if info.get('dram_thru', 0) > 0 or info.get('mem_thru', 0) > 0:
                    return True
    return False
