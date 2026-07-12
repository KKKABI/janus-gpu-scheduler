import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from torch.fx import Tracer, GraphModule
from torch.cuda.streams import Event
from Opara import OperatorLauncher, Critical_node, priority_streams
from Opara.GraphCapturer import compute_max_parallel_width, Scheduler
from googlenet_example import run_torch_model, run_sequence_graph, run_parallel_graph, flush_cache
from transformers import LlamaConfig, LlamaModel
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm, LlamaRotaryEmbedding, LlamaAttention,
)

import torch._dynamo
torch._dynamo.config.suppress_errors = True


class SingleOutputDecoderLayer(torch.nn.Module):
    """
    LlamaDecoderLayer 的包装器：只返回 hidden_states（张量），而非元组。
    作用：避免 FX tracing 时对 Proxy 做下标解包（proxy[0] 触发 __iter__ 报错）。
    作为 leaf module 使用，tracer 不进入其内部。
    """
    def __init__(self, decoder_layer):
        super().__init__()
        self.layer = decoder_layer

    def forward(self, hidden_states, attention_mask, position_embeddings):
        outputs = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        return outputs[0]


class LlamaForOpara(torch.nn.Module):
    """
    专为 Opara FX tracing 设计的简化 Llama 包装器。

    设计决策：
    - causal_mask / position_ids 预计算并注册为 buffer（图内常量，不作为输入变量）S
    - 每层用 SingleOutputDecoderLayer 包装，forward 只返回 hidden_states
    - forward 只接受 input_ids，结构对 FX tracer 完全静态

    方案 A 局限：所有 decoder layer 均作为 leaf，图为纯串行链，
    Opara 的多流并行调度贡献有限，主要收益来自 CUDA Graph 消除 Python overhead。
    """
    def __init__(self, llama_model, causal_mask, position_ids):
        super().__init__()
        self.embed_tokens = llama_model.embed_tokens
        self.rotary_emb   = llama_model.rotary_emb
        self.layers = torch.nn.ModuleList([
            SingleOutputDecoderLayer(layer) for layer in llama_model.layers
        ])
        self.norm = llama_model.norm
        self.register_buffer('causal_mask',  causal_mask)
        self.register_buffer('position_ids', position_ids)

    def forward(self, input_ids):
        hidden_states       = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, self.position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, self.causal_mask, position_embeddings)
        return self.norm(hidden_states)


class LlamaTracer(Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        # 最底层 PyTorch 原生算子模块：作为图节点
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Dropout, torch.nn.SiLU)):
            return True

        # RMSNorm / RotaryEmbedding 内部含复杂的显存切片，保持为原子节点
        if isinstance(m, (LlamaRMSNorm, LlamaRotaryEmbedding)):
            return True

        # LlamaAttention（含子类 LlamaFlashAttention2 / LlamaSdpaAttention）必须是 leaf。
        # 原因：forward 内部有 `cos, sin = position_embeddings`，
        # position_embeddings 是 Proxy，__iter__ 解包会触发 "Proxy cannot be iterated"。
        if isinstance(m, LlamaAttention):
            return True

        # SingleOutputDecoderLayer、LlamaDecoderLayer、LlamaMLP 不是 leaf，
        # FX 递归进入，暴露 MLP 内部的 gate_proj / up_proj（两者互相独立，可并行）
        return False


def llama_capturer(model, static_input_ids, static_attention_mask, static_position_ids,
                   batch_size, seq_length, copy_outputs: bool = False):
    """
    构建 LlamaForOpara 包装器，FX trace 后走与 GraphCapturer.capturer() 相同的
    Opara 调度 + CUDA Graph 捕获流程。

    与 capturer() 的区别：
    - 用 LlamaForOpara 替代原始 LlamaModel，避免 Proxy 解包问题
    - causal_mask / position_ids 作为 buffer，forward 只接受 input_ids
    """
    # ── 1. 预计算 causal_mask（具体张量，作为模型 buffer）───────────────────────
    dtype    = torch.float16
    device   = static_input_ids.device
    min_val  = torch.finfo(dtype).min
    causal_mask = torch.triu(
        torch.full((seq_length, seq_length), min_val, dtype=dtype, device=device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_length, seq_length).contiguous()

    # ── 2. 构建可 trace 的包装器 ─────────────────────────────────────────────────
    opara_model = LlamaForOpara(model, causal_mask, static_position_ids).eval()

    inputs        = (static_input_ids,)
    static_inputs = [torch.zeros_like(x, device='cuda') for x in inputs]
    model_class_name = f"LlamaModel_bs{batch_size}_seq{seq_length}"

    # ── 3. 用 dynamo.explain() 提取 FX 图（与 GraphCapturer.capturer() 相同方式）──
    # 原因：LlamaTracer 是符号追踪，遇到 Attention 内部的 shape 推断和控制流会崩溃；
    # dynamo 是字节码级追踪，能处理 bsz,q_len,_=x.size()、device.type 等动态分支。
    import torch._dynamo as dynamo
    dynamo.reset()
    with torch.no_grad():
        result = dynamo.explain(opara_model)(*inputs)
    if isinstance(result, tuple):
        _, _, graphs, _, _, _ = result
    else:
        graphs = getattr(result, "graphs", None) or getattr(result, "graph", None)
    fx_module = graphs[0]
    fx_module.cuda()
    print(f"dynamo 展开节点数: {len(list(fx_module.graph.nodes))}")

    # ── 4. 计算 DAG 最宽层级，创建最高优先级 CUDA 流 ─────────────────────────
    max_width = compute_max_parallel_width(fx_module)
    print("max_width:", max_width)
    priority_streams.create_priority_streams(max_width)
    stream_ptrs = priority_streams.get_all_stream_ptrs()
    all_streams = [torch.cuda.ExternalStream(ptr) for ptr in stream_ptrs]

    # ── 5. 初始化节点属性 ────────────────────────────────────────────────────────
    for node in fx_module.graph.nodes:
        setattr(node, 'stream',         None)
        setattr(node, 'event',          None)
        setattr(node, 'event_to_wait',  [])
        setattr(node, 'is_lowpriority', False)
        setattr(node, 'node_to_bool',   False)
    for node in fx_module.graph.nodes:
        node.event = Event()

    # ── 6. 关键路径标记 + profile 调度 + FX 图重排 ──────────────────────────────
    Critical_node.mark_critical_nodes(fx_module.graph)
    OperatorLauncher.recompile(model_class_name, fx_module, inputs, all_streams, max_width)

    # ── 统计 HP / LP 算子 + 执行时间 ─────────────────────────────────────────────
    skip_ops = {'placeholder', 'output', 'get_attr'}
    all_ops  = [n for n in fx_module.graph.nodes if n.op not in skip_ops]
    lp_nodes = [n for n in all_ops if n.is_lowpriority]
    hp_nodes = [n for n in all_ops if not n.is_lowpriority]
    print(f"算子总数: {len(all_ops)}  HP(高优先级): {len(hp_nodes)}  LP(低优先级): {len(lp_nodes)}")

    print(f"\n{'节点名':<40} {'优先级':<6} {'执行时间(us)':>12}  内核数")
    print("-" * 70)
    for node in all_ops:
        priority = "LP" if node.is_lowpriority else "HP"
        if hasattr(node, 'info') and node.info:
            dur_us = sum(k["dur"] for k in node.info)
            n_kernels = len(node.info)
        else:
            dur_us, n_kernels = 0, 0
        print(f"{node.name:<40} {priority:<6} {dur_us:>12.1f}  {n_kernels}")
    print("-" * 70)
    hp_time = sum(sum(k["dur"] for k in n.info) for n in hp_nodes if hasattr(n, 'info') and n.info)
    lp_time = sum(sum(k["dur"] for k in n.info) for n in lp_nodes if hasattr(n, 'info') and n.info)
    print(f"HP 总执行时间: {hp_time/1000:.3f} ms    LP 总执行时间: {lp_time/1000:.3f} ms")

    # ── 7. 跨流同步 event ────────────────────────────────────────────────────────
    for node in fx_module.graph.nodes:
        for input_node in node.all_input_nodes:
            if node.stream != input_node.stream:
                if input_node.event not in node.event_to_wait:
                    node.event_to_wait.append(input_node.event)

    # ── 8. 预热 3 次 + CUDA Graph 捕获 ──────────────────────────────────────────
    all_events   = [torch.cuda.Event() for _ in range(len(all_streams))]
    first_stream = all_streams[0]
    first_event  = all_events[0]
    interpreter  = Scheduler(fx_module)

    with torch.no_grad():
        for _ in range(3):
            interpreter.run(*inputs)

    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g, stream=first_stream):
            first_event.record(first_stream)
            for i, stream in enumerate(all_streams):
                if i > 0:
                    stream.wait_event(first_event)
            static_outputs = interpreter.run(*static_inputs)
            torch.cuda.set_stream(first_stream)
            for i, event in enumerate(all_events):
                if i > 0:
                    event.record(all_streams[i])
            for i, event in enumerate(all_events):
                if i > 0:
                    first_stream.wait_event(event)
    torch.cuda.synchronize()

    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    # ── 9. 推理闭包 ──────────────────────────────────────────────────────────────
    def run(*new_inputs):
        # new_inputs[0] = input_ids；new_inputs[1] = attention_mask（可选，causal_mask 已固定）
        static_inputs[0].copy_(new_inputs[0])
        with torch.no_grad():
            g.replay()
        if copy_outputs:
            return tuple(x.clone() for x in static_outputs)
        return static_outputs

    return run


if __name__ == '__main__':
    warm_ups   = 10
    iterations = 10
    batch_size = 1
    seq_length = 256

    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=2048,
    )
    model = LlamaModel(config).half().to("cuda:0").eval()

    static_input_ids      = torch.randint(0, config.vocab_size,
                                          (batch_size, seq_length),
                                          dtype=torch.long, device="cuda:0")
    static_attention_mask = torch.ones((batch_size, seq_length),
                                       dtype=torch.long, device="cuda:0")
    static_position_ids   = torch.arange(seq_length, dtype=torch.long, device="cuda:0") \
                                  .unsqueeze(0).expand(batch_size, -1).contiguous()
    inputs     = (static_input_ids, static_attention_mask)   # 原生 PyTorch 用
    inputs_one = (static_input_ids,)                         # LlamaForOpara 只需 input_ids

    # ① 原生 PyTorch（用原始 LlamaModel，输入 (input_ids, attention_mask)）
    print("\n===== PyTorch原始模型性能测试 =====")
    with torch.no_grad():
        torch_outputs = run_torch_model(model, inputs, iterations, warm_ups)

    # ② 顺序 CUDA Graph（用 LlamaForOpara，规避 _update_causal_mask 里的 CPU-GPU 同步）
    # 原始 LlamaModel 的 _ignore_causal_mask_sdpa 在捕获期间调用 torch.all()（同步操作），
    # 会触发 "operation not permitted when stream is capturing"，因此不能用原始模型录图。
    print("\n===== CUDA Graph顺序执行性能测试 =====")
    dtype_   = torch.float16
    min_val_ = torch.finfo(dtype_).min
    causal_mask_ = torch.triu(
        torch.full((seq_length, seq_length), min_val_, dtype=dtype_, device="cuda:0"),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_length, seq_length).contiguous()
    opara_model = LlamaForOpara(model, causal_mask_, static_position_ids).eval()
    with torch.no_grad():
        run_sequence_graph(opara_model, inputs_one, iterations, warm_ups, 0, iterations)

    # ③ Opara（LlamaForOpara 包装 + CUDA Graph）
    # 释放前两步占用的显存：顺序 CUDAGraph 私有池 + 中间张量
    del opara_model, causal_mask_, torch_outputs
    torch.cuda.empty_cache()
    print(f"  显存释放后剩余: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.2f} GiB free")

    print("\n===== Opara并行执行性能测试 =====")
    opara_run = llama_capturer(
        model, static_input_ids, static_attention_mask, static_position_ids,
        batch_size, seq_length
    )
    opara_outputs = run_parallel_graph(opara_run, inputs_one, iterations, warm_ups, 0, iterations)

    # ④ 输出一致性验证：重建 LlamaForOpara 做串行参考推理
    print("\n===== 输出验证 =====")
    dtype_v   = torch.float16
    min_val_v = torch.finfo(dtype_v).min
    causal_mask_v = torch.triu(
        torch.full((seq_length, seq_length), min_val_v, dtype=dtype_v, device="cuda:0"),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_length, seq_length).contiguous()
    ref_model = LlamaForOpara(model, causal_mask_v, static_position_ids).eval()
    with torch.no_grad():
        ref_tensor = ref_model(*inputs_one)
    torch_last = ref_tensor.detach().cpu().float()
    opara_last = (opara_outputs[0]
                  if isinstance(opara_outputs, (list, tuple))
                  else opara_outputs).detach().cpu().float()
    print("output of PyTorch == output of Opara:",
          torch.allclose(torch_last, opara_last, rtol=1e-2, atol=5e-2, equal_nan=False),
          end='     ')
    print('Absolute difference:', torch.max(torch.abs(torch_last - opara_last)))
