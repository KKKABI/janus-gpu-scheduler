import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
from Opara import GraphCapturer
from googlenet_example import run_torch_model, run_sequence_graph, run_parallel_graph, flush_cache

if __name__ == '__main__':
    # 配置参数 (VGG模型可以支持更大的batch)
    warm_ups = 100
    iterations = 300
    batch_size = 1  # VGG结构简单，可以使用较大batch
    input_size = 224  # VGG的标准输入尺寸
    
    # 创建输入张量
    x = torch.randint(low=0, high=256, size=(batch_size, 3, input_size, input_size), 
                     dtype=torch.float32).to(device="cuda:0")
    
    # 创建VGG16模型 (可以选择其他变体)
    model = torchvision.models.vgg16(pretrained=True)
    inputs = (x,)
    model = model.to(device="cuda:0").eval()
    
    # 测试原生PyTorch性能
    print("\n" + "="*70)
    print("Testing Native PyTorch VGG16")
    print("="*70)
    y = run_torch_model(model, inputs, iterations, warm_ups)
    
    # 测试顺序CUDA图性能
    print("\n" + "="*70)
    print("Testing Sequential CUDA Graph")
    print("="*70)
    run_sequence_graph(model, inputs, iterations, warm_ups, 0, 300)
    
    # 使用Opara捕获模型并测试并行性能
    print("\n" + "="*70)
    print("Testing Opara Parallel Execution")
    print("="*70)
    Opara = GraphCapturer.capturer(inputs, model)
    output = run_parallel_graph(Opara, inputs, iterations, warm_ups, 0, 300)
    
    # 验证输出一致性
    res = output[0]
    if res.dtype == torch.float16:
        res = res.float()
    
    is_close = torch.allclose(y, res, rtol=1e-05, atol=1e-05, equal_nan=False)
    max_diff = torch.max(torch.abs(y.detach() - res.detach()))
    
    print("\nValidation Results:")
    print(f"Output of PyTorch == Output of Opara: {is_close}")
    print(f"Absolute difference: {max_diff.item():.6f}")
    
    # 内存使用报告
    print("\nMemory Usage Report:")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024 / 1024:.2f} MB")
    
    # VGG特有分析：全连接层占比
    fc_params = sum(p.numel() for name, p in model.named_parameters() if 'classifier' in name)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nVGG Architecture Analysis:")
    print(f"Total parameters: {total_params:,}")
    print(f"FC parameters: {fc_params:,} ({fc_params/total_params:.1%})")
    print("="*70)