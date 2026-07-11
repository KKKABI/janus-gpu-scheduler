from functools import reduce
from operator import mul
from torch.cuda.streams import Stream, Event
from Opara import ModelProfiler
import sys
import math
from collections import deque
from Opara.Scheduler import Scheduler, OperatorTask, KernelProfile, ResourceModel, VirtualSM
import os
path = os.path.abspath(os.path.dirname(__file__))
output_file_path = path + '/profile_result/output.txt'
output_file = open(output_file_path, "w")

def compute_node_slack(fx_nodes):
    """
    使用 profile duration 计算每个节点的 slack（松弛时间）。

    原理：
    - earliest_start[node] = max(所有前驱的 earliest_start + duration)
    - latest_finish[node]  = min(所有后继的 latest_finish - duration)
    - slack[node] = latest_finish - earliest_start - duration

    slack > 0 表示该节点可以在不延长 DAG 总执行时间的前提下推迟调度。
    """
    import torch

    def get_duration(node):
        if not hasattr(node, 'info') or not node.info:
            return 0.0
        return sum(info['dur'] for info in node.info) / 1000.0  # 单位: μs

    topo_order = list(fx_nodes)

    # Forward pass: 最早开始时间
    earliest_start = {}
    for node in topo_order:
        max_pred_end = 0.0
        for pred in node.all_input_nodes:
            if isinstance(pred, torch.fx.Node):
                pred_end = earliest_start.get(pred, 0.0) + get_duration(pred)
                max_pred_end = max(max_pred_end, pred_end)
        earliest_start[node] = max_pred_end

    total_time = max(
        (earliest_start.get(node, 0.0) + get_duration(node))
        for node in topo_order
    ) if topo_order else 0.0

    # Backward pass: 最晚结束时间
    latest_finish = {}
    for node in reversed(topo_order):
        min_succ_start = total_time
        for succ in node.users:
            if isinstance(succ, torch.fx.Node):
                succ_latest_start = latest_finish.get(succ, total_time) - get_duration(succ)
                min_succ_start = min(min_succ_start, succ_latest_start)
        latest_finish[node] = min_succ_start

    # Slack 计算
    slack = {}
    for node in topo_order:
        dur = get_duration(node)
        est = earliest_start.get(node, 0.0)
        lft = latest_finish.get(node, total_time)
        slack[node] = max(0.0, lft - est - dur)

    return slack


def pop_lowPriorty_from_queue(queue, slack=None, tau=0.5, max_light_dur=100.0, slack_ratio=0.3):
    """根据 Co-location Suitability Score (CSS) 从队列中弹出低优先级算子。

    计算方法（对队列中具有 kernel 信息的节点集合 V）：
    - T_op_raw: 节点所有 kernel 的 duration 求和（使用原始单位，避免提前缩放）
    - D_th/D_reg/D_mem: 分别由每个 kernel 的线程数、寄存器总数、共享内存总数求和
    - 对上述每个维度先取自然对数再做 min-max 归一化
    - S_time = 归一化后的 ln(T_op_raw)
    - S_res = (归一化 ln(D_th) + 归一化 ln(D_reg) + 归一化 ln(D_mem)) / 3
    - CSS = 0.5 * S_time + 0.5 * S_res

    如果节点在关键路径上（`is_critical` 为 True），则始终保持高优先级。
    """
    lowpriority = []
    candidates = []

    # 收集候选节点的原始数值统计
    for node in list(queue):
        if not hasattr(node, 'info') or len(node.info) == 0:
            continue
        if getattr(node, 'is_critical', False):
            continue

        T_op_raw = 0.0
        D_th = 0.0
        D_reg = 0.0
        D_mem = 0.0

        for info in node.info:
            # duration: use raw dur (trace uses microseconds); protect against missing
            dur = info.get('dur', 0.0)
            T_op_raw += max(dur, 0.0)

            args = info.get('args', {})
            block = args.get('block', (1, 1, 1))
            grid = args.get('grid', (1, 1, 1))
            block_threads = int(block[0]) * int(block[1]) * int(block[2])
            blocks = int(grid[0]) * int(grid[1]) * int(grid[2])
            threads = block_threads * blocks
            D_th += threads

            reg_per_thread = args.get('registers per thread', 0)
            D_reg += reg_per_thread * threads

            shared_mem_per_block = args.get('shared memory', 0)
            D_mem += shared_mem_per_block * blocks

        # 防止全为 0 导致 log(0)
        eps = 1e-6
        T_op_raw = max(T_op_raw, eps)
        D_th = max(D_th, eps)
        D_reg = max(D_reg, eps)
        D_mem = max(D_mem, eps)

        candidates.append({
            'node': node,
            'T_op_raw': T_op_raw,
            'D_th': D_th,
            'D_reg': D_reg,
            'D_mem': D_mem,
        })

    if not candidates:
        return lowpriority

    # 计算 ln 并做 min-max 归一化
    ln_T = [math.log(c['T_op_raw']) for c in candidates]
    ln_th = [math.log(c['D_th']) for c in candidates]
    ln_reg = [math.log(c['D_reg']) for c in candidates]
    ln_mem = [math.log(c['D_mem']) for c in candidates]

    def normalize(values):
        vmin = min(values)
        vmax = max(values)
        if abs(vmax - vmin) < 1e-12:
            return [0.5 for _ in values]
        return [(v - vmin) / (vmax - vmin) for v in values]

    norm_T = normalize(ln_T)
    norm_th = normalize(ln_th)
    norm_reg = normalize(ln_reg)
    norm_mem = normalize(ln_mem)

    # 计算 CSS，分级处理
    for idx, c in enumerate(candidates):
        S_time = norm_T[idx]
        S_res = (norm_th[idx] + norm_reg[idx] + norm_mem[idx]) / 3.0

        CSS = 0.5 * S_time + 0.5 * S_res

        if CSS < tau:
            node = c['node']
            duration = c['T_op_raw']  # 原始执行时长（μs）

            if duration < max_light_dur:
                # LP-light: 短算子，跟 HP 同轮捡漏
                node.is_lowpriority = True
                try:
                    queue.remove(node)
                except ValueError:
                    pass
                lowpriority.append(node)
            elif slack is not None and slack.get(node, 0.0) > duration * slack_ratio:
                # LP-heavy: 大算子且有足够的 slack → defer 到 Phase 2
                node.is_lp_heavy = True
                # 不 remove，由 launch() 统一管理
            # else: LP-heavy 但 slack 不足 → 留在队列当 HP 处理

    return lowpriority





def launch(nodes , in_degree, sharedMemPerMultiprocessor, regsPerMultiprocessor, maxThreadsPerMultiprocessor, numSms , all_streams ,max_width, slack=None):

    sm_specs = {
        'shared_mem_total': sharedMemPerMultiprocessor,
        'register_total': regsPerMultiprocessor,
        'warp_total': maxThreadsPerMultiprocessor//32
    }
    resource_model = ResourceModel(sm_count=numSms, sm_specs=sm_specs)
    scheduler = Scheduler(resource_model)
    current_time = 0.0


    # 从 FX node 构建 KernelProfile 列表
    def kernel_profiles_from_node(node):
        return [
            KernelProfile(
                name=info["name"],
                duration=info["dur"] / 1000.0,
                shared_mem=info["args"].get("shared memory", 0),
                registers=info["args"].get("registers per thread", 0) *
                          info["args"]["block"][0] * info["args"]["block"][1] * info["args"]["block"][2],
                warps=(info["args"]["block"][0] * info["args"]["block"][1] * info["args"]["block"][2] + 32 - 1) // 32,
                blocks=info["args"]["grid"][0] * info["args"]["grid"][1] * info["args"]["grid"][2]
            )
            for info in node.info
        ]
           # 初始化队列
    queue = deque()
    prestage_ops = []
    result = []
    deferred_heavy = []  # LP-heavy 节点暂存到 Phase 2

    def enqueue_ready_nodes(queue):
        ready_ops = []
        for node in queue:
            kernels = kernel_profiles_from_node(node)
            ready_ops.append(OperatorTask(node.name, kernels))
        return ready_ops

    for node ,deg in  in_degree.items():
        if deg == 0:
            queue.append(node)

    # ========== Phase 1: HP + LP-light ==========
    while queue:
        # 1. 提取 LP-light；LP-heavy 被标记 is_lp_heavy 但不出队列
        lowpriority_ops = pop_lowPriorty_from_queue(queue, slack)

        # 2. 将 LP-heavy 从队列中取出，暂存到 Phase 2
        lp_heavy_round = []
        remaining = deque()
        for node in queue:
            if getattr(node, 'is_lp_heavy', False):
                node.is_lowpriority = False  # 确保不被当作 LP-light 处理
                lp_heavy_round.append(node)
            else:
                remaining.append(node)
        queue = remaining
        deferred_heavy.extend(lp_heavy_round)

        # 如果没有 HP 或 LP-light 留下，退出 Phase 1
        if not queue:
            break

        # 3. 调度 HP（非 lowpriority、非 heavy 的节点）
        ready_ops = enqueue_ready_nodes(queue)
        scheduled_ops = scheduler.schedule(ready_ops, current_time)

        allocate_streams = []
        for op in scheduled_ops:
            node = nodes[op.name]
            for input_node in node.all_input_nodes:
                if input_node.name in prestage_ops and input_node.node_to_bool is False:
                    result.append(op.name)
                    node.stream = input_node.stream
                    allocate_streams.append(node.stream)
                    input_node.node_to_bool = True
                    break
            for pre_op in prestage_ops:
                pre_node = nodes[pre_op]
                if node.stream != pre_node.stream:
                    node.event_to_wait.append(pre_node.event)
        for op in scheduled_ops:
            node = nodes[op.name]
            if node.stream is None:
                for i in range(max_width):
                    if all_streams[i] not in allocate_streams:
                        result.append(op.name)
                        node.stream = all_streams[i]
                        allocate_streams.append(node.stream)
                        break
            for pre_op in prestage_ops:
                pre_node = nodes[pre_op]
                if node.stream != pre_node.stream:
                    node.event_to_wait.append(pre_node.event)
        prestage_ops.clear()
        for op in scheduled_ops:
            prestage_ops.append(op.name)

        # 4. LP-light 分配普通流
        for node in lowpriority_ops:
            result.append(node.name)
            for input_node in node.all_input_nodes:
                if input_node.node_to_bool is False and input_node.is_lowpriority is True:
                    node.stream = input_node.stream
                    input_node.node_to_bool = True
                    break
            if node.stream is None:
                node.stream = Stream()
                all_streams.append(node.stream)

        # 5. 构建已调度节点列表（用于 queue 维护 + in_degree）
        scheduled_node_names = []
        for op in scheduled_ops:
            scheduled_node_names.append(op.name)
        for node in lowpriority_ops:
            scheduled_node_names.append(node.name)

        # 从 queue 中移除已调度的节点
        queue = deque(n for n in queue if n.name not in scheduled_node_names)

        # 更新 in_degree（HP + LP-light）
        for name in scheduled_node_names:
            for user in nodes[name].users:
                in_degree[user] -= 1
                if in_degree[user] == 0:
                    queue.append(user)

        # 更新 in_degree（LP-heavy 算作"已调度"以解开后继依赖）
        for node in lp_heavy_round:
            for user in nodes[node.name].users:
                in_degree[user] -= 1
                if in_degree[user] == 0:
                    queue.append(user)

        # current_time = resource_model.run_until_next_launchable()

    # ========== Phase 2: LP-heavy 调度 ==========
    if deferred_heavy:
        queue = deque(deferred_heavy)
        prestage_ops.clear()

        while queue:
            ready_ops = enqueue_ready_nodes(queue)
            scheduled_ops = scheduler.schedule(ready_ops, current_time)

            if not scheduled_ops:
                break

            # Stream 分配（复用 priority stream）
            allocate_streams = []
            for op in scheduled_ops:
                node = nodes[op.name]
                for input_node in node.all_input_nodes:
                    if input_node.name in prestage_ops and input_node.node_to_bool is False:
                        node.stream = input_node.stream
                        allocate_streams.append(node.stream)
                        input_node.node_to_bool = True
                        break
                for pre_op in prestage_ops:
                    pre_node = nodes[pre_op]
                    if node.stream != pre_node.stream:
                        node.event_to_wait.append(pre_node.event)

            for op in scheduled_ops:
                node = nodes[op.name]
                if node.stream is None:
                    for i in range(max_width):
                        if all_streams[i] not in allocate_streams:
                            node.stream = all_streams[i]
                            allocate_streams.append(node.stream)
                            break
                    if node.stream is None:
                        # 所有 priority stream 已被占用，新建普通流
                        node.stream = Stream()
                        all_streams.append(node.stream)
                for pre_op in prestage_ops:
                    pre_node = nodes[pre_op]
                    if node.stream != pre_node.stream:
                        node.event_to_wait.append(pre_node.event)

            prestage_ops.clear()
            for op in scheduled_ops:
                prestage_ops.append(op.name)

            # 加入 result（排在 Phase 1 所有节点之后）
            for op in scheduled_ops:
                result.append(op.name)

            # 从 queue 中移除已调度的节点
            scheduled_names = [op.name for op in scheduled_ops]
            queue = deque(n for n in queue if n.name not in scheduled_names)

            # 更新 in_degree
            for name in scheduled_names:
                for user in nodes[name].users:
                    in_degree[user] -= 1
                    if in_degree[user] == 0:
                        queue.append(user)

    return result
    
    


       
   
import json
import os

def get_resource_from_json(path):
    sharedMemPerMultiprocessor = 0
    regsPerMultiprocessor = 0
    maxThreadsPerMultiprocessor = 0
    numSms = 0
    with open(path) as f:
        data = json.load(f)

    try:
        sharedMemPerMultiprocessor = data['deviceProperties'][0]['sharedMemPerMultiprocessor']
        regsPerMultiprocessor = data['deviceProperties'][0]['regsPerMultiprocessor']
        maxThreadsPerMultiprocessor = data['deviceProperties'][0]['maxThreadsPerMultiprocessor']
        numSms = data['deviceProperties'][0]["numSms"]
    except (KeyError, IndexError):
        pass  # 使用默认值

    step_num = 0
    for event in data["traceEvents"]:
        if "torch/fx/interpreter.py(97): run" in event["name"] and "run_node" not in event["name"]:
            step_num += 1
    # print("step_num", step_num)
    # 获取run_node事件、kernel_launch事件、kernel事件
    run_node_events = []
    kernel_launch_events = []
    kernel_events = []
    for event in data["traceEvents"]:
        if "run_node" in event["name"]:
            run_node_events.append(event)

        if event["name"] == "cudaLaunchKernel":
            kernel_launch_events.append(event)

        if event.get("cat", "None") == "kernel":
            kernel_events.append(event)


    # 计算获取一个step中的run_node事件、kernel_launch事件、kernel事件
    if step_num == 0:
        # 如果没有步骤，可能是profile数据格式不同，返回空或默认值
        return [], sharedMemPerMultiprocessor, regsPerMultiprocessor, maxThreadsPerMultiprocessor, numSms
    one_step_range_of_node = len(run_node_events) // step_num
    one_step_range_of_kernel_launch = len(kernel_launch_events) // step_num
    one_step_range_of_kernel = len(kernel_events) // step_num
    start = step_num - 1
    end = step_num
    run_node_events = run_node_events[start*one_step_range_of_node:end*one_step_range_of_node]
    kernel_launch_events = kernel_launch_events[start*one_step_range_of_kernel_launch:end*one_step_range_of_kernel_launch]
    kernel_events = kernel_events[start*one_step_range_of_kernel:end*one_step_range_of_kernel]


    # 根据时间轴范围获取由node事件触发的kernel_launch事件
    node2kernels = []
    kernel_num = 0
    for i, node_event in enumerate(run_node_events):
        node2kernels.append([])
        for j, kernel_launch_event in enumerate(kernel_launch_events):
            if node_event["ts"] <= kernel_launch_event["ts"] and node_event["ts"] + node_event["dur"] >= kernel_launch_event["ts"]:
                node2kernels[i].append(kernel_events[j])
                kernel_num += 1

    sharedMemPerMultiprocessor = data['deviceProperties'][0]['sharedMemPerMultiprocessor']
    regsPerMultiprocessor = data['deviceProperties'][0]['regsPerMultiprocessor']
    maxThreadsPerMultiprocessor = data['deviceProperties'][0]['maxThreadsPerMultiprocessor']
    numSms = data['deviceProperties'][0]["numSms"]


    return node2kernels, sharedMemPerMultiprocessor, regsPerMultiprocessor, maxThreadsPerMultiprocessor , numSms



         
# def get_topo(fx_nodes):
#     nodes = {node.name: node for node in fx_nodes}
#     in_degree = {node.name: 0 for node in nodes.values()}
#     for node in nodes.values():
#         for input_node in node.all_input_nodes:
#             in_degree[node.name] += 1
#     visited = set()
#     return in_degree, nodes

def get_topo(fx_nodes):
    nodes = {node.name: node for node in fx_nodes}
    in_degree = {node: 0 for node in fx_nodes }
    for node in fx_nodes:
        for input_node in node.all_input_nodes:
           in_degree[node] += 1
    return nodes , in_degree

def recompile(model_class_name, graph_module, inputs, all_streams, max_width):
    
    path = os.path.abspath(os.path.dirname(__file__))
    # model_class_name = graph_module.__class__.__name__
    for i in inputs:
        model_class_name += "_" + str(i.shape)
    path += "/profile_result/" + model_class_name + ".pt.trace.json"

    if os.path.exists(path) is False:
        ModelProfiler.profile_serial(graph_module, inputs, path)
    node2kernels, sharedMemPerMultiprocessor, regsPerMultiprocessor, maxThreadsPerMultiprocessor , numSms = get_resource_from_json(path)

    for i, node in enumerate(graph_module.graph.nodes):
        if not hasattr(node, 'info'):
            if i < len(node2kernels):
                setattr(node, 'info', node2kernels[i])
            else:
                setattr(node, 'info', [])  # 默认空列表，如果没有kernel信息
       

    torch_nodes , in_degree = get_topo(graph_module.graph.nodes)

    # 计算 DAG 松弛时间，用于 LP-heavy defer 决策
    slack = compute_node_slack(graph_module.graph.nodes)

    result = launch(torch_nodes , in_degree, sharedMemPerMultiprocessor, regsPerMultiprocessor, maxThreadsPerMultiprocessor, numSms , all_streams, max_width, slack=slack)
    
    # for stream in all_streams:
    #     print(stream)
        
    size = len(result)
    for i in range(size - 1):
        torch_nodes[result[i]].append(torch_nodes[result[i+1]])
    
    graph_module.graph.lint()
    graph_module.recompile()

    # with open('output.txt', 'w') as f:
    #     sys.stdout = f
    #     for node in graph_module.graph.nodes:
    #         if node.info:  
    #             names = [kernel["name"] for kernel in node.info]
    #             print(';'.join(names))
    #             # print(node.endtime)  
    #     sys.stdout = sys.__stdout__
    

    # path = os.path.abspath(os.path.dirname(__file__))    
    
    # path += "/profile_result/" + model_class_name + "_parallel_CudaGraph(1)" +".pt.trace.json"

    # # if os.path.exists(path) is False:
    # ModelProfiler.profile_parallel_cudagraph(graph_module, inputs, path , all_streams)

    # get_endtime_from_json(path , graph_module.graph.nodes)
    # in_degree, torch_nodes = get_topo(graph_module.graph.nodes)
    # launch_2(torch_nodes , in_degree)

    # for node in graph_module.graph.nodes:
    #     print(node.need_record)

    
    

    with open('output.txt', 'w') as f:
        sys.stdout = f
        for node in graph_module.graph.nodes:
            if node.info:  
                names = [kernel["name"] for kernel in node.info]
                print(';'.join(names))  
        sys.stdout = sys.__stdout__



    # with open('node2kernels.txt', 'w') as f:
    #     sys.stdout = f
    #     count = 0
    #     num = 0
    #     for item in node2kernels:
    #         if len(item) > 1:
    #            count = count + 1
    #         if len(item) > num:
    #             num = len(item)
    #         print(item)
    #     print("num = ", num)
    #     print("count = ", count)
    # sys.stdout = sys.__stdout__
    
