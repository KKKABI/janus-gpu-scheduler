import torch
from torch.fx import Interpreter


class Executor(Interpreter):

    def __init__(self, module , all_streams):
        super().__init__(module)
        self.streams = all_streams
       

    def run_node(self, n):
        for event in n.event_to_wait:
            n.stream.wait_event(event)
        torch.cuda.set_stream(stream=n.stream)
        
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        
        self.env[n] = getattr(self, n.op)(n.target, args, kwargs)
        n.event.record(n.stream)
        return self.env[n]
    
    def run(self, *args):
        self.env = {}
        self.args_iter = iter(args)
        # events = [torch.cuda.Event() for _ in range(len(self.streams))]
        event = torch.cuda.Event()
        for i , stream in enumerate(self.streams):
            if i != len(self.streams) - 1:
                self.streams[i].wait_event(event)
        print("zyf-----------")        
        for node in self.module.graph.nodes:
            self.env[node] = self.run_node(node)
            if node.op == 'output':
                output_val = self.env[node]
                event.record(self.streams[-1])
                print(event)
                return output_val

def profile_serial(symbolic_traced, inputs, path):
    interpreter = Interpreter(symbolic_traced)
    
    def trace_handler(p):
        p.export_chrome_trace(path)
    
    with torch.profiler.profile(
        on_trace_ready=trace_handler,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
    ) as p:
        for i in range(1):
            out_torch = interpreter.run(*inputs)
            p.step()
    return


def profile_parallel_cudagraph(symbolic_traced, inputs, path ,all_streams):

  



    interpreter = Executor(symbolic_traced , all_streams)

    out_torch = interpreter.run(*inputs)
    
    def trace_handler(p):
        p.export_chrome_trace(path)
    
    with torch.profiler.profile(
        on_trace_ready=trace_handler,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
    ) as p:
        for i in range(1):
            out_torch = interpreter.run(*inputs)
            p.step()
    return 


    