import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from Opara import GraphCapturer
from googlenet_example import run_torch_model, run_sequence_graph, run_parallel_graph, flush_cache
from transformers import LlamaConfig, LlamaModel

# 设置 TorchDynamo 出错时回退到 eager 模式
import torch._dynamo
torch._dynamo.config.suppress_errors = True

if __name__ == '__main__':
    # 可配置参数
    warm_ups = 10
    iterations = 10  # Llama较大，默认100次
    batch_size = 1
    seq_length = 1024

    # 配置Llama模型
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=2048,
        use_cache=False,  # 禁用 KV 缓存，确保每次 forward 独立
    )
    model = LlamaModel(config).half().to("cuda:0").eval()
    

    # 构造输入
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), dtype=torch.long).to("cuda:0")
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to("cuda:0")
    inputs = (input_ids, attention_mask)

    # 原始PyTorch性能测试
    print("\n===== PyTorch原始模型性能测试 =====")
    with torch.no_grad():
        torch_outputs = run_torch_model(model, inputs, iterations, warm_ups)

    # CUDA Graph顺序执行性能测试
    print("\n===== CUDA Graph顺序执行性能测试 =====")
    with torch.no_grad():
        run_sequence_graph(model, inputs, iterations, warm_ups, 0, iterations)

    # Opara并行执行性能测试
    print("\n===== Opara并行执行性能测试 =====")
    with torch.no_grad():
        Opara = GraphCapturer.capturer(inputs, model)
        opara_outputs = run_parallel_graph(Opara, inputs, iterations, warm_ups, 0, iterations)

    # 输出一致性验证
    print("\n===== 输出验证 =====")
    torch_last_hidden = torch_outputs.last_hidden_state.detach().cpu() if hasattr(torch_outputs, 'last_hidden_state') else torch_outputs[0].detach().cpu()
    if isinstance(opara_outputs, (list, tuple)):
        opara_last_hidden = opara_outputs[0].detach().cpu()
    else:
        opara_last_hidden = opara_outputs.detach().cpu()
    print("output of PyTorch == output of Opara:", torch.allclose(torch_last_hidden, opara_last_hidden, rtol=1e-05, atol=1e-05, equal_nan=False), end='     ')
    print('Absolute difference:', torch.max(torch.abs(torch_last_hidden - opara_last_hidden)))
