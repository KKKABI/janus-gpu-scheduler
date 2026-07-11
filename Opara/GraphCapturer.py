import torch
from torch.fx import Interpreter
import torch._dynamo as dynamo
from Opara import OperatorLauncher
# from Opara import StreamAllocator
from torch._functorch.partitioners import draw_graph
from collections import defaultdict,deque
from torch.cuda.streams import Stream, Event
from Opara import priority_streams
import os
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
        
        self.env[n] = getattr(self, n.op)(n.target, args, kwargs)

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
            


def compute_max_parallel_width(fx_module: torch.fx.GraphModule) -> int:
    graph = fx_module.graph
    node_to_users = defaultdict(set)
    node_to_deps = defaultdict(set)

    # Map nodes to their dependencies and users
    for node in graph.nodes:
        for arg in node.all_input_nodes:
            node_to_users[arg].add(node)
            node_to_deps[node].add(arg)

    # Topological level tracking
    in_degree = {node: len(node_to_deps[node]) for node in graph.nodes}
    level_count = defaultdict(int)
    queue = deque()

    # Start with nodes with zero dependencies
    for node in graph.nodes:
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
            


def capturer(inputs, model, copy_outputs: bool = False):
    assert isinstance(inputs, (list, tuple)), f"inputs is of type {type(inputs)} instead of list"
    static_inputs = [torch.zeros_like(x, device='cuda') for x in inputs]

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
    # print(fx_module.graph, file=output_file)
    fx_module.cuda()
    model_class_name = model.__class__.__name__
    
    max_width = compute_max_parallel_width(fx_module)

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
            
    OperatorLauncher.recompile(model_class_name, fx_module, inputs, all_streams , max_width)

    print(stream for stream in all_streams)
        
    for node in graph.nodes:
        for input_node in node.all_input_nodes:
            if node.stream != input_node.stream:
                if input_node.event not in node.event_to_wait:
                    node.event_to_wait.append(input_node.event)

    
   
    all_events = [torch.cuda.Event() for _ in range(len(all_streams))]
    first_stream = all_streams[0]
    first_event = all_events[0]
    interpreter = Scheduler(fx_module)

    # with torch.autocast(device_type='cuda', dtype=torch.float16):

    with torch.no_grad():
        for i in range(3):
            interpreter.run(*inputs)
    with torch.no_grad():
        # capture
        g = torch.cuda.CUDAGraph()

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

    def run(*new_inputs):
        assert isinstance(new_inputs, (list, tuple)), f"inputs is of type {type(new_inputs)} instead of list"
        assert len(static_inputs) == len(new_inputs), f"{len(static_inputs)} == {len(new_inputs)}"
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)  # cuda graph can only read data from the same address
        with torch.no_grad():
            g.replay()
        if copy_outputs:
            return [x.clone() for x in static_outputs]
        else:
            return static_outputs

    return run
