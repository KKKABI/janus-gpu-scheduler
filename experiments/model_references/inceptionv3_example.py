import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
from Opara import GraphCapturer
from googlenet_example import run_torch_model, run_sequence_graph, run_parallel_graph, flush_cache

if __name__ == '__main__':
    # 配置参数
    warm_ups = 100
    iterations = 300
    batch_size = 1  # Inception-v3需要更多内存，使用较小batch
    input_size = 299  # Inception-v3的标准输入尺寸
    
    # 创建输入张量
    x = torch.randint(low=0, high=256, size=(batch_size, 3, input_size, input_size), 
                     dtype=torch.float32).to(device="cuda:0")
    
    # 创建Inception-v3模型 (注意输出格式处理)
    model = torchvision.models.inception_v3(pretrained=True)
    inputs = (x,)
    model = model.to(device="cuda:0").eval()
    
    # 测试原生PyTorch性能
    # print("\n" + "="*70)
    # print("Testing Native PyTorch Inception-v3")
    # print("="*70)
    # y = run_torch_model(model, inputs, iterations, warm_ups)
    
    # 测试顺序CUDA图性能
    # print("\n" + "="*70)
    # print("Testing Sequential CUDA Graph")
    # print("="*70)
    # run_sequence_graph(model, inputs, iterations, warm_ups, 0, 300)
    
    # 使用Opara捕获模型并测试并行性能
    print("\n" + "="*70)
    print("Testing Opara Parallel Execution")
    print("="*70)
    Opara = GraphCapturer.capturer(inputs, model)
    output = run_parallel_graph(Opara, inputs, iterations, warm_ups, 0, 300)
    
    # Inception-v3输出处理 (返回主输出和辅助输出)
    res = output[0] if isinstance(output, tuple) else output
    
    # 验证输出一致性 (只比较主输出)
    # y_main = y[0] if isinstance(y, tuple) else y
    
    # if res.dtype == torch.float16:
    #     res = res.float()
    
    # is_close = torch.allclose(y_main, res, rtol=1e-05, atol=1e-05, equal_nan=False)
    # max_diff = torch.max(torch.abs(y_main.detach() - res.detach()))
    
    # print("\nValidation Results:")
    # print(f"Output of PyTorch == Output of Opara: {is_close}")
    # print(f"Absolute difference: {max_diff.item():.6f}")
    
    # # 内存使用报告
    # print("\nMemory Usage Report:")
    # print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    # print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024 / 1024:.2f} MB")
    # print("="*70)