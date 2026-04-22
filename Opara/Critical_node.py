import collections
import torch

def mark_critical_nodes(graph):

    for node in graph.nodes:
        
        setattr(node, 'is_critical', False)
    # 计算每个节点的正向深度（从输入节点到该节点的最长路径长度）
    in_degree = {}
    d1 = {}  # 正向深度
    # 初始化入度和正向深度
    for node in graph.nodes:
        # 获取节点的所有输入节点（前驱节点）
        inputs = [inp for inp in node.all_input_nodes if isinstance(inp, torch.fx.Node)]
        in_degree[node] = len(inputs)
        d1[node] = 0  # 初始化为0
    
    # 使用队列进行拓扑排序（BFS）
    queue = collections.deque()
    # 入度为0的节点（输入节点）加入队列
    for node in graph.nodes:
        if in_degree[node] == 0:
            queue.append(node)
            d1[node] = 1  # 输入节点深度为1
    
    # 正向传播计算深度
    while queue:
        u = queue.popleft()
        for v in u.users:
            if not isinstance(v, torch.fx.Node):
                continue
            # 更新后继节点的最大深度
            d1[v] = max(d1[v], d1[u] + 1) if v in d1 else d1[u] + 1
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # 找到输出节点
    output_node = None
    for node in graph.nodes:
        if node.op == 'output':
            output_node = node
            break
    if output_node is None:
        output_node = list(graph.nodes)[-1]  # 默认取最后一个节点
    
    # 计算反向深度（从节点到输出节点的最长路径长度）
    d2 = {}  # 反向深度
    # 初始化出度
    out_degree = {}
    for node in graph.nodes:
        users = [user for user in node.users if isinstance(user, torch.fx.Node)]
        out_degree[node] = len(users)
        d2[node] = 0
    
    # 从输出节点开始反向传播
    queue = collections.deque([output_node])
    d2[output_node] = 1  # 输出节点深度为1
    
    # 反向拓扑排序（使用节点的输入节点作为前驱）
    while queue:
        u = queue.popleft()
        # 获取当前节点的所有输入节点（前驱）
        for v in u.all_input_nodes:
            if not isinstance(v, torch.fx.Node):
                continue
            # 更新前驱节点的最大反向深度
            d2[v] = max(d2[v], d2[u] + 1) if v in d2 else d2[u] + 1
            out_degree[v] -= 1
            if out_degree[v] == 0:
                queue.append(v)
    
    # 计算关键路径长度
    critical_path_length = 0
    for node in graph.nodes:
        if node in d1 and node in d2:
            path_len = d1[node] + d2[node] - 1
            if path_len > critical_path_length:
                critical_path_length = path_len
    
    # 标记关键节点
    for node in graph.nodes:
        if node in d1 and node in d2:
            if d1[node] + d2[node] - 1 == critical_path_length:
                node.is_critical = True
            else:
                node.is_critical = False
        else:
            node.is_critical = False