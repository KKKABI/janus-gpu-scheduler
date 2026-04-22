import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from Opara import GraphCapturer
from googlenet_example import run_torch_model, run_sequence_graph, run_parallel_graph, flush_cache

if __name__ == '__main__':
    warm_ups = 30
    iterations = 100
    
    # 使用torchvision中的MobileNetV2模型
    from torchvision.models import mobilenet_v2
    x = torch.randint(low=0, high=256, size=(1, 3, 224, 224), dtype=torch.float32).to(device="cuda:0")
    
    # 创建MobileNetV2模型并加载预训练权重
    model = mobilenet_v2(weights='DEFAULT')
    inputs = (x,)
    model = model.to(device="cuda:0").eval()
    
    # 打印模型结构信息
    print("="*80)
    print(f"Testing MobileNetV2 with input size {x.shape}")
    print("="*80)
    
    # 运行原生PyTorch基准测试
    print("\n[1/3] Running native PyTorch benchmark...")
    y = run_torch_model(model, inputs, iterations, warm_ups)
    
    # 运行序列化CUDA图基准测试
    print("\n[2/3] Running sequential CUDA Graph benchmark...")
    run_sequence_graph(model, inputs, iterations, warm_ups, 0, iterations)
    
    # 捕获并运行并行化图
    print("\n[3/3] Running Opara parallel graph...")
    Opara = GraphCapturer.capturer(inputs, model)
    output = run_parallel_graph(Opara, inputs, iterations, warm_ups, 0, iterations)
    
    # 验证结果一致性
    res = output[0]
    if res.dtype == torch.float16:
        res = res.float()
    
    # 详细的结果验证
    abs_diff = torch.abs(y.detach() - res.detach())
    max_diff = torch.max(abs_diff)
    mean_diff = torch.mean(abs_diff)
    
    print("\n" + "="*80)
    print("Validation Results:")
    print("="*80)
    print(f"Output consistency: {torch.allclose(y, res, rtol=1e-05, atol=1e-05, equal_nan=False)}")
    print(f"Max absolute difference: {max_diff.item():.6f}")
    print(f"Mean absolute difference: {mean_diff.item():.6f}")
    
    # 显存分析
    torch.cuda.reset_peak_memory_stats()
    _ = model(x)
    native_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    torch.cuda.reset_peak_memory_stats()
    _ = Opara(x)
    opara_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    print("\n" + "="*80)
    print("Memory Usage Comparison:")
    print("="*80)
    print(f"Native PyTorch: {native_mem:.2f} MB")
    print(f"Opara: {opara_mem:.2f} MB")
    print(f"Memory reduction: {((native_mem - opara_mem) / native_mem * 100):.1f}%")
    
    # 额外信息
    print("\n" + "="*80)
    print("Additional Information:")
    print("="*80)
    print(f"Total iterations: {iterations}")
    print(f"Warm-up iterations: {warm_ups}")
    print(f"Input shape: {x.shape}")
    print(f"Device: {x.device}")