import os
import sys
# 将仓库根目录加入 Python 路径，使 `from Opara import ...` 能正常工作
# 因此必须从仓库根目录运行，不能 cd 进 examples/ 再执行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.cuda.nvtx as nvtx
import torch
import torchvision
import numpy as np
from Opara import GraphCapturer


# ─────────────────────────────────────────────
# 基准测试函数①：原生 PyTorch 推理
# ─────────────────────────────────────────────
def run_torch_model(model, inputs, iterations, warm_ups):
    # 用 CUDA Event 计时，比 Python time 精确（直接在 GPU 端打时间戳）
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    with torch.no_grad():
        # 预热：让 GPU 进入稳定状态，预热结果不计入统计
        for _ in range(warm_ups):
            model(*inputs)
        for i in range(iterations):
            flush_cache()               # 清 GPU L2 cache，避免缓存热效应使结果偏快
            torch.cuda._sleep(1_000_000)  # 让 GPU 短暂空闲，模拟真实推理间隔
            start_events[i].record()
            y = model(*inputs)
            end_events[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        std = np.std(times)
        output_str = ('Time of native PyTorch:', str(np.mean(times)) + ' ms', "std: " + str(std))
        print('{:<30} {:<20} {:<20}'.format(*output_str))
    return y


# ─────────────────────────────────────────────
# 基准测试函数②：顺序 CUDA Graph（无并行化）
# ─────────────────────────────────────────────
# 对比意义：比原生 PyTorch 快，是因为消除了 Python/CUDA API 调用开销
# 但算子仍然串行执行，用来证明 Opara 的加速来自"并行化本身"而非仅靠 CUDA Graph
def run_sequence_graph(symbolic_traced, inputs, iterations, warm_ups, start_index, end_index):
    with torch.no_grad():
        for i in range(warm_ups):
            symbolic_traced(*inputs)    # 预热

    with torch.no_grad():
        g1 = torch.cuda.CUDAGraph()
        # 录制：把顺序推理过程捕获成 CUDA Graph
        with torch.cuda.graph(g1):
            out_torch_graph_without_stream = symbolic_traced(*inputs)

        time_list = []
        torch.cuda.synchronize()
        for i in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            g1.replay()                 # 回放：不重新执行 Python，直接重放 GPU 命令序列
            end.record()
            end.synchronize()
            tim = start.elapsed_time(end)
            time_list.append(tim)
        average_time = np.mean(time_list[start_index: end_index])
        std = np.std(time_list[start_index: end_index])
        output_str = ('Time of sequential CUDA Graph:', str(average_time) + ' ms', "std: " + str(std))
        print('{:<30} {:<20} {:<20}'.format(*output_str))
        return out_torch_graph_without_stream


# ─────────────────────────────────────────────
# 基准测试函数③：Opara 并行推理
# ─────────────────────────────────────────────
# 参数 Opara 是 GraphCapturer.capturer() 返回的 run() 闭包
# 调用它等价于将新输入拷贝到静态缓冲区后执行 g.replay()
def run_parallel_graph(Opara, inputs, iterations, warm_ups, start_index, end_index):
    time_list = []

    for i in range(iterations):
        flush_cache()                   # 同样清 cache，保证测量公平
        torch.cuda._sleep(1_000_000)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = Opara(*inputs)
        end.record()
        end.synchronize()
        tim = start.elapsed_time(end)
        time_list.append(tim)
    average_time = np.mean(time_list[start_index: end_index])
    std = np.std(time_list[start_index: end_index])
    output_str = ('Time of Opara:', str(average_time) + ' ms', "std: " + str(std) )
    print('{:<30} {:<20} {:<20}'.format(*output_str))
    return output


# ─────────────────────────────────────────────
# Cache 刷新工具
# ─────────────────────────────────────────────
# 分配 4MB GPU 张量，每次测试前用 zero_() 填满，把 GPU L2 cache 里的旧数据挤掉
# 这是 GPU benchmark 的标准做法，防止前一次推理的数据留在 cache 里使后续测量偏快
cache = torch.empty(int(4 * (1024 ** 2)), dtype=torch.int8, device='cuda')
def flush_cache():
    cache.zero_()


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")

if __name__ == '__main__':
    warm_ups = 10
    iterations = 300
    x = torch.randint(low=0, high=256, size=(1, 3, 224, 224), dtype=torch.float32).to(device="cuda:0")
    model = torchvision.models.googlenet().eval()

    inputs = (x,)
    model = model.to(device="cuda:0").eval()

    # ① 原生 PyTorch
    y = run_torch_model(model, inputs, iterations, warm_ups)

    # ② 顺序 CUDA Graph（对比基准）
    run_sequence_graph(model, inputs, iterations, warm_ups, 0, 300)

    # ③ Opara：capturer() 只调用一次，内部完成 profiling → 调度 → CUDA Graph 捕获，比较耗时
    #    返回的 Opara 是轻量的 run() 闭包，后续每次推理都只是 g.replay()
    Opara = GraphCapturer.capturer(inputs, model)
    output = run_parallel_graph(Opara, inputs, iterations, warm_ups, 0, 300)

    # 正确性验证：并行化后的输出与原始 PyTorch 结果数值上应一致（允许 1e-5 误差）
    res = output[0]
    if res.dtype == torch.float16:
        res = res.float()
    print("output of PyTorch == output of Opara:", torch.allclose(y, res,rtol=1e-05,atol=1e-05,equal_nan =False), end='     ')
    print('Absolute difference:', torch.max(torch.abs(y.detach() - res.detach())))
