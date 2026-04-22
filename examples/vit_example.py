import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from Opara import GraphCapturer
from googlenet_example import run_torch_model, run_sequence_graph, run_parallel_graph, flush_cache

if __name__ == '__main__':
    warm_ups = 100
    iterations = 300
    
    # 使用torchvision中的ViT模型
    from torchvision.models import vit_b_16
    x = torch.randint(low=0, high=256, size=(1, 3, 224, 224), dtype=torch.float32).to(device="cuda:0")
    
    # 创建ViT模型并加载预训练权重
    model = vit_b_16(weights='DEFAULT')
    inputs = (x,)
    model = model.to(device="cuda:0").eval()
    
    # 运行原生PyTorch基准测试
    y = run_torch_model(model, inputs, iterations, warm_ups)
    
    # 运行序列化CUDA图基准测试
    run_sequence_graph(model, inputs, iterations, warm_ups, 0, 300)
    
    # 捕获并运行并行化图
    Opara = GraphCapturer.capturer(inputs, model)
    output = run_parallel_graph(Opara, inputs, iterations, warm_ups, 0, 300)
    
    # 验证结果一致性
    res = output[0]
    if res.dtype == torch.float16:
        res = res.float()
        
    print("output of PyTorch == output of Opara:", 
          torch.allclose(y, res, rtol=1e-05, atol=1e-05, equal_nan=False),
          end=' ')
    print('Absolute difference:', torch.max(torch.abs(y.detach() - res.detach())))
    
    # 报告显存使用情况
    print("Memory used by PyTorch:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")