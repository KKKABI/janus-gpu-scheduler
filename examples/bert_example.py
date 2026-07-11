import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from Opara import GraphCapturer
from googlenet_example import run_torch_model, run_sequence_graph, run_parallel_graph

# 添加本地模型路径
sys.path.append('/public_0/ZYF/model/bert-base')
from transformers import BertModel

def flush_cache():
    """清空GPU缓存确保公平比较"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    # 可配置参数
    warm_ups = 10
    iterations = 300
    batch_size = 1
    seq_length = 16

    # 加载bert-base模型
    model = BertModel.from_pretrained('/public_0/ZYF/model/bert-base').eval()
    model = model.to("cuda:0")

    # 准备BERT输入 - 使用原始随机生成方式
    input_ids = torch.randint(0, 30000, (batch_size, seq_length), dtype=torch.long).to("cuda:0")
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to("cuda:0")
    inputs = (input_ids, attention_mask)
    
    # 原始PyTorch性能测试
    # print("\n===== PyTorch原始模型性能测试 =====")
    # with torch.no_grad():
    #     torch_outputs = run_torch_model(model, inputs, iterations, warm_ups)
    
    # CUDA Graph顺序执行性能测试
    # print("\n===== CUDA Graph顺序执行性能测试 =====")
    # with torch.no_grad():
    #     run_sequence_graph(model, inputs, iterations, warm_ups, 0, 300)
    
    # Opara并行执行性能测试
    print("\n===== Opara并行执行性能测试 =====")
    with torch.no_grad():
        Opara = GraphCapturer.capturer(inputs, model)
        opara_outputs = run_parallel_graph(Opara, inputs, iterations, warm_ups, 0, 300)
    
    # 验证输出一致性
    # print("\n===== 输出验证 =====")
    
    # # 提取PyTorch输出
    # torch_last_hidden = torch_outputs.last_hidden_state.detach().cpu()
    
    # # 处理Opara输出
    # if isinstance(opara_outputs, (list, tuple)):
    #     # Opara输出是元组，第一个元素是last_hidden_state
    #     opara_last_hidden = opara_outputs[0].detach().cpu()
    # else:
    #     # Opara输出是单个张量
    #     opara_last_hidden = opara_outputs.detach().cpu()
    
    # # 检查形状
    # if torch_last_hidden.shape != opara_last_hidden.shape:
    #     print(f"形状不匹配! PyTorch: {torch_last_hidden.shape}, Opara: {opara_last_hidden.shape}")
    # else:
    #     # 计算差异
    #     diff = torch.abs(torch_last_hidden - opara_last_hidden)
    #     max_diff = torch.max(diff).item()
    #     mean_diff = torch.mean(diff).item()
        
    #     print(f"最大绝对误差: {max_diff:.6f}")
    #     print(f"平均绝对误差: {mean_diff:.6f}")
    #     print(f"输出是否在容差范围内(1e-5): {max_diff < 1e-5}")
    
    # # 显存使用报告
    # print("\n===== 显存使用报告 =====")
    # print("PyTorch最大显存使用:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
    
    # # 打印可配置参数
    # print("\n===== 测试配置 =====")
    # print(f"批次大小: {batch_size}")
    # print(f"序列长度: {seq_length}")
    # print(f"预热次数: {warm_ups}")
    # print(f"测试迭代: {iterations}")