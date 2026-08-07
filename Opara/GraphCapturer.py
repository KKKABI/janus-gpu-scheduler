import torch
from torch.fx import Interpreter
import torch._dynamo as dynamo
from Opara import OperatorLauncher
from Opara import Critical_node
# from Opara import StreamAllocator
from torch._functorch.partitioners import draw_graph
from collections import defaultdict,deque
from torch.cuda.streams import Stream, Event
from Opara import priority_streams
import json
import os
from pathlib import Path
import sys
path = os.path.abspath(os.path.dirname(__file__))
output_file_path = path + '/profile_result/output.txt'
output_file = open(output_file_path, "w")

class Scheduler(Interpreter):

    

    def run_node(self, n):
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        if n.event_to_wait:
            for event in n.event_to_wait:
                # print(n.name, n.stream)
                n.stream.wait_event(event)
       
        torch.cuda.set_stream(stream=n.stream)

        args, kwargs = self.fetch_args_kwargs_from_env(n)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        
        q3_capture_phase = (
            bool(os.environ.get("OPARA_Q3_PROFILE_MAP"))
            and getattr(self, "_q3_phase", None) == "graph_capture"
        )
        if q3_capture_phase:
            torch.cuda.nvtx.range_push(f"FX::GRAPH_CAPTURE::{n.name}")
        try:
            self.env[n] = getattr(self, n.op)(n.target, args, kwargs)
        finally:
            if q3_capture_phase:
                torch.cuda.nvtx.range_pop()

        # if n.need_record == True:        
        #     n.event.record(n.stream)
        # n.event.record(n.stream)

        is_record = False
        for user in n.users:
            if n.stream != user.stream:
                if is_record is False:
                    n.event.record(n.stream)
                    is_record = True
        return self.env[n]
    
  
    
    def run(self, *args):
       
        self.env = {}
        self.args_iter = iter(args)   
      

        for node in self.module.graph.nodes:        
            
            self.env[node] = self.run_node(node)
            
            if node.op == 'output':
                output_val = self.env[node]
                return output_val
            


def compute_max_parallel_width(fx_module: torch.fx.GraphModule, excluded_ops=None) -> int:
    graph = fx_module.graph
    excluded_ops = set(excluded_ops or ())
    active_nodes = [node for node in graph.nodes if node.op not in excluded_ops]
    active_set = set(active_nodes)
    node_to_users = defaultdict(set)
    node_to_deps = defaultdict(set)

    # Map nodes to their dependencies and users
    for node in active_nodes:
        for arg in node.all_input_nodes:
            if arg not in active_set: continue
            node_to_users[arg].add(node)
            node_to_deps[node].add(arg)

    # Topological level tracking
    in_degree = {node: len(node_to_deps[node]) for node in active_nodes}
    level_count = defaultdict(int)
    queue = deque()

    # Start with nodes with zero dependencies
    for node in active_nodes:
        if in_degree[node] == 0:
            queue.append((node, 0))
            level_count[0] += 1

    max_width = 0

    while queue:
        current_node, level = queue.popleft()
        max_width = max(max_width, level_count[level])

        for user in node_to_users[current_node]:
            in_degree[user] -= 1
            if in_degree[user] == 0:
                queue.append((user, level + 1))
                level_count[level + 1] += 1

    return max_width
            


def capturer(inputs, model, copy_outputs: bool = False, alpha=0.9, selection_mode='cosine', time_domain=True, capture_backend='dynamo_explain'):
    assert isinstance(inputs, (list, tuple)), f"inputs is of type {type(inputs)} instead of list"
    static_inputs = [torch.zeros_like(x, device='cuda') for x in inputs]

    if capture_backend == 'make_fx':
        from torch.fx.experimental.proxy_tensor import make_fx
        with torch.no_grad():
            fx_module = make_fx(model)(*inputs)
    elif capture_backend == 'dynamo_explain':
        dynamo.reset()
        with torch.no_grad():
            result = dynamo.explain(model)(*inputs)
            if isinstance(result, tuple):
                explanation, out_guards, graphs, ops_per_graph, break_reasons, explanation_verbose = result
            else:
                explanation = getattr(result, "explanation", None)
                out_guards = getattr(result, "out_guards", None)
                graphs = getattr(result, "graphs", None) or getattr(result, "graph", None)
                ops_per_graph = getattr(result, "ops_per_graph", None)
                break_reasons = getattr(result, "break_reasons", None)
                explanation_verbose = getattr(result, "explanation_verbose", None)
        fx_module = graphs[0]
    else:
        raise ValueError(f"unsupported capture backend: {capture_backend}")
    # print(fx_module.graph, file=output_file)
    fx_module.cuda()
    model_class_name = model.__class__.__name__
    
    excluded_ops = {'placeholder', 'get_attr'} if capture_backend == 'make_fx' else set()
    max_width = compute_max_parallel_width(fx_module, excluded_ops)

    print("max_width :" , max_width)
   
    priority_streams.create_priority_streams(max_width)
    
    # 优先级流
    stream_ptrs = priority_streams.get_all_stream_ptrs()
    all_streams = [torch.cuda.ExternalStream(ptr) for ptr in stream_ptrs]

    # # 直接创建普通 CUDA 流，不使用 priority_streams 模块
    #all_streams = [torch.cuda.Stream() for _ in range(max_width)]

    
    graph = fx_module.graph

    for node in graph.nodes:
        setattr(node, 'stream', None)
        setattr(node, 'event', None)       
        setattr(node, 'event_to_wait', [])
        setattr(node, 'is_lowpriority', False)
        setattr(node, 'node_to_bool', False)

    for node in graph.nodes:
        node.event = Event()

    Critical_node.mark_critical_nodes(graph)
    OperatorLauncher.recompile(model_class_name, fx_module, inputs, all_streams, max_width, alpha, selection_mode, time_domain, exclude_metadata=bool(excluded_ops))

    # print(stream for stream in all_streams)
        
    for node in graph.nodes:
        for input_node in node.all_input_nodes:
            if node.stream != input_node.stream:
                if input_node.event not in node.event_to_wait:
                    node.event_to_wait.append(input_node.event)

    q3_profile_map = os.environ.get("OPARA_Q3_PROFILE_MAP")
    if q3_profile_map:
        node_rows = []
        for node in graph.nodes:
            stream_ptr = None
            stream_index = None
            if node.stream is not None:
                stream_ptr = int(node.stream.cuda_stream)
                stream_index = next(
                    (
                        index for index, stream in enumerate(all_streams)
                        if int(stream.cuda_stream) == stream_ptr
                    ),
                    None,
                )
            kernels = []
            for info in getattr(node, "info", []) or []:
                info_args = info.get("args", {})
                kernels.append({
                    "name": str(info.get("name", "")),
                    "duration_us": float(info.get("dur", 0.0)),
                    "grid": list(info_args.get("grid", ())),
                    "block": list(info_args.get("block", ())),
                })
            node_rows.append({
                "name": node.name,
                "op": node.op,
                "stream_index": stream_index,
                "stream_ptr": stream_ptr,
                "wait_event_count": len(node.event_to_wait),
                "kernels": kernels,
            })
        Path(q3_profile_map).write_text(
            json.dumps({"nodes": node_rows}, indent=2), encoding="utf-8"
        )

    
   
    all_events = [torch.cuda.Event() for _ in range(len(all_streams))]
    first_stream = all_streams[0]
    first_event = all_events[0]
    interpreter = Scheduler(fx_module)
    interpreter._q3_phase = "warmup"

    # with torch.autocast(device_type='cuda', dtype=torch.float16):

    with torch.no_grad():
        for i in range(3):
            interpreter.run(*inputs)
    with torch.no_grad():
        # capture
        g = torch.cuda.CUDAGraph()

        interpreter._q3_phase = "graph_capture"
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
        interpreter._q3_phase = "idle"

        if not isinstance(static_outputs, (list, tuple)):
            static_outputs = (static_outputs,)

    def run(*new_inputs):
        assert isinstance(new_inputs, (list, tuple)), f"inputs is of type {type(new_inputs)} instead of list"
        assert len(static_inputs) == len(new_inputs), f"{len(static_inputs)} == {len(new_inputs)}"
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)  # cuda graph can only read data from the same address
        with torch.no_grad():
            g.replay()
        if copy_outputs:
            from torch.utils._pytree import tree_map
            return tree_map(lambda value: value.clone() if isinstance(value, torch.Tensor) else value, static_outputs)
        else:
            return static_outputs

    run._opara_graph = g
    run._opara_graph_module = fx_module
    run._opara_streams = all_streams
    return run
