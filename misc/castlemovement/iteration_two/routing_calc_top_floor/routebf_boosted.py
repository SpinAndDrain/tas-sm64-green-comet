import time
from functools import lru_cache


class Node():
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class Graph():
    def __init__(self):
        self.nodes = {}

    def set_visit_count(self, node, count):
        # do nothing if requested visit count is simply one
        if count <= 1:
            return
        
        # create duplicated nodes to maintain graph integrity 
        duplicates = [Node(f'{node.name}{i}') for i in range(1, count + 1)]

        # copy edges of original node to all duplicates
        for d in duplicates:
            self[d] = self[node].copy()

        # swap out all edges from any node to the original node with the new duplicate nodes
        for n in self.nodes:
            edges = self[n]
            if node not in edges:
                continue
            for d in duplicates:
                if d == n:
                    continue
                # set new edges
                edges[d] = edges[node]
            # delete old original edge
            del edges[node]
        
        # delete original node
        del self[node]

    def __repr__(self):
        return '\n'.join([f'{str(node)} -> {", ".join([f"{e}({w})" for e, w in edges.items()])}' for node, edges in self.nodes.items()])

    def __getitem__(self, key):
        return self.nodes.get(key, {})
    
    def __setitem__(self, key, value):
        self.nodes[key] = value
    
    def __delitem__(self, key):
        del self.nodes[key]


def find_routes_with_memo(graph, start_node, end_node):
    best_route = None
    best_cost = float('inf')

    # Preprocess edge lists (optional)
    sorted_edges = {node: sorted(edges.items(), key=lambda x: x[1]) for node, edges in graph.nodes.items()}
    
    visited_count = len(graph.nodes)

    # Convert node objects to unique ids or hashable strings
    node_ids = {node: i for i, node in enumerate(graph.nodes)}

    @lru_cache(maxsize=None)
    def dfs(current_node_id, visited_mask):
        # Zustand der besten Route speichern
        current_node = list(graph.nodes.keys())[current_node_id]

        # Wenn alle Knoten besucht sind, returniere den Kostenwert 0
        if visited_mask == (1 << visited_count) - 1:
            return 0, []  # Keine weiteren Kosten, leere Route

        best = float('inf')
        best_path = []

        for neighbor, weight in sorted_edges[current_node]:
            nid = node_ids[neighbor]
            if (visited_mask >> nid) & 1:
                continue  # Already visited, prunieren

            new_mask = visited_mask | (1 << nid)
            cost, path = dfs(nid, new_mask)

            # Wenn wir einen besseren Weg gefunden haben
            if cost + weight < best:
                best = cost + weight
                best_path = [neighbor] + path

        return best, best_path

    start_id = node_ids[start_node]
    min_cost, best_path = dfs(start_id, 1 << start_id)

    # Rückgabe der besten Route und der minimalen Kosten
    return [start_node] + best_path, min_cost



def find_routes(graph, start_node, end_node):
    # Keep track of the best route and costs so far.
    best_route = None
    best_weight = float('inf')

    aborted = 0
    updated = 0

    # Sort all edges of each node to find optimal solutions more quickly.
    sorted_edges = { node: sorted(edges, key=edges.get) for node, edges in graph.nodes.items() }

    def dfs(current_node, path, current_weight):
        nonlocal best_route, best_weight, sorted_edges, aborted, updated

        # In case the current route is already slower than the currently
        # fastest known route, abort exploring this route further to save
        # time.
        if current_weight >= best_weight:
            aborted += 1
            return

        # Add the visiting node to the current path.
        path.append(current_node)

        # Given that in the normalized graph all nodes must be visited exactly once,
        # a route is only valid if the length of the path is equal to the number of
        # nodes in the graph. If this is the case, a route that is faster than the
        # previous one has been found and is updated here.
        #print(f'{len(path)} <=> {len(graph.nodes)}')
        if len(path) == len(graph.nodes):
            best_route = list(path)
            best_weight = current_weight
            updated += 1
        else:
            # Determine all possible remaining nodes in the current state, then start
            # a recursive dfs for each node successively.
            # A node is only a remaining node in the current state if it was not
            # visited before. If no remaining nodes are left, the end node is
            # automatically used to ensure it is always visited last.
            remaining_nodes = [node for node in sorted_edges[current_node] if node not in path and node != end_node]
            if not remaining_nodes:
                remaining_nodes = [end_node]
            #print(f'{path} => {remaining_nodes}')
            for neighbor in remaining_nodes:
                dfs(neighbor, path, current_weight + graph[current_node][neighbor])
        
        # Backtracking
        path.pop()

    dfs(start_node, [], 0)

    print(f'Aborted: {aborted}\nUpdated: {updated}\nTotal: {aborted+updated}')

    return best_route, best_weight


def main(start_id):

    # Create all nodes
    TFF = Node('TFF')
    TFB = Node('TFB')
    SL = Node('SL')
    DOD = Node('DOD')
    TTM = Node('TTM')
    TBI = Node('TBI')
    RC = Node('RC')

    start = TFF if start_id == 0 else TFB
    end = RC

    # connect
    g = Graph()
    if start is TFF:
        g[TFF] = {SL:262,DOD:220,TTM:216,TBI:264}
    else:
        g[TFB] = {SL:432,DOD:250,TTM:102,TBI:404}
    g[SL] = {DOD:400,TTM:408,TBI:512,RC:594}
    g[DOD] = {SL:290,TTM:206,TBI:260,RC:328}
    g[TTM] = {SL:312,DOD:196,TBI:284,RC:366}
    g[TBI] = {SL:238,DOD:216,TTM:226,RC:362}
    g[RC] = {}

    ##################################################################

    s = time.time()
    route, length = find_routes_with_memo(g, start, end)
    e = time.time()

    print(f"Fastest route from {start} to {end} is [{length}]: {' -> '.join([str(n) for n in route])}")
    print(f"Took {(e-s):.2f}s")


if __name__ == "__main__":
    main(0)
    main(1)
