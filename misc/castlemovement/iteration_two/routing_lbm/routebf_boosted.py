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


def main(end_id):

    # Create all nodes
    SPAWN = Node('SPAWN')
    VCUM = Node('VCUM')
    BoF = Node('BoF')
    WoF = Node('WoF')
    JRB = Node('JRB')
    CCM = Node('CCM')
    AM = Node('AM')
    B1DW = Node('B1DW')
    WCCT = Node('WCCT')
    PSS = Node('PSS')
    HMC = Node('HMC')
    LLL = Node('LLL')
    SSL = Node('SSL')
    DDD = Node('DDD')
    B2FS = Node('B2FS')
    TFF = Node('TFF')
    TFB = Node('TFB')

    start = SPAWN
    end = TFF if end_id == 0 else TFB

    # Connect all edges
    g = Graph()
    
    if end is TFF:
        g[SPAWN] = {VCUM:370,BoF:719,WoF:707,JRB:705,CCM:707,AM:715,B1DW:705,WCCT:597,PSS:701,HMC:1258,LLL:1134,SSL:1222,DDD:1104,B2FS:1120}
        g[VCUM] = {BoF:483,WoF:445,JRB:461,CCM:445,AM:453,B1DW:443,WCCT:309,PSS:439,HMC:1122,LLL:928,SSL:960,DDD:842,B2FS:858,TFF:355}
        g[BoF] = {VCUM:586,WoF:324,JRB:310,CCM:330,AM:300,B1DW:372,WCCT:320,PSS:316,HMC:929,LLL:735,SSL:767,DDD:649,B2FS:665,TFF:278}
        g[WoF] = {VCUM:586,BoF:412,JRB:420,CCM:418,AM:430,B1DW:440,WCCT:432,PSS:456,HMC:1061,LLL:867,SSL:899,DDD:781,B2FS:797,TFF:398}
        g[JRB] = {VCUM:586,BoF:318,WoF:322,CCM:312,AM:320,B1DW:338,WCCT:342,PSS:354,HMC:949,LLL:755,SSL:787,DDD:669,B2FS:685,TFF:292}
        g[CCM] = {VCUM:586,BoF:314,WoF:322,JRB:334,AM:328,B1DW:358,WCCT:322,PSS:340,HMC:959,LLL:765,SSL:797,DDD:679,B2FS:695,TFF:266}
        g[AM] = {VCUM:586,BoF:603,WoF:565,JRB:581,CCM:565,B1DW:563,WCCT:429,PSS:559,HMC:1190,LLL:996,SSL:1028,DDD:910,B2FS:926,TFF:475}
        g[B1DW] = {VCUM:586,BoF:388,WoF:354,JRB:372,CCM:354,AM:382,WCCT:379,PSS:372,HMC:1003,LLL:809,SSL:841,DDD:723,B2FS:739,TFF:352}
        g[WCCT] = {VCUM:453,BoF:228,WoF:190,JRB:206,CCM:190,AM:198,B1DW:188,PSS:184,HMC:867,LLL:673,SSL:705,DDD:587,B2FS:603,TFF:100}
        g[PSS] = {VCUM:453,BoF:228,WoF:190,JRB:206,CCM:190,AM:198,B1DW:188,WCCT:54,HMC:867,LLL:673,SSL:705,DDD:587,B2FS:603,TFF:100,PSS:184}
        g[HMC] = {VCUM:540,BoF:315,WoF:277,JRB:293,CCM:277,AM:285,B1DW:275,WCCT:141,PSS:271,LLL:156,SSL:192,DDD:386,B2FS:402,TFF:187}
        g[LLL] = {VCUM:586,BoF:677,WoF:639,JRB:655,CCM:639,AM:647,B1DW:637,WCCT:503,PSS:633,HMC:300,SSL:166,DDD:516,B2FS:532,TFF:549}
        g[SSL] = {VCUM:586,BoF:681,WoF:643,JRB:659,CCM:643,AM:651,B1DW:641,WCCT:507,PSS:637,HMC:412,LLL:158,DDD:622,B2FS:638,TFF:553}
        g[DDD] = {VCUM:586,BoF:538,WoF:500,JRB:516,CCM:500,AM:508,B1DW:498,WCCT:364,PSS:494,HMC:542,LLL:348,SSL:382,B2FS:118,TFF:410}
        g[B2FS] = {VCUM:586,BoF:490,WoF:452,JRB:468,CCM:452,AM:460,B1DW:450,WCCT:316,PSS:446,HMC:504,LLL:310,SSL:344,DDD:114,TFF:362}
        g[TFF] = {}
    else:
        g[SPAWN] = {VCUM:370,BoF:719,WoF:707,JRB:705,CCM:707,AM:715,B1DW:705,WCCT:597,PSS:701,HMC:1258,LLL:1134,SSL:1222,DDD:1104,B2FS:1120}
        g[VCUM] = {BoF:483,WoF:445,JRB:461,CCM:445,AM:453,B1DW:443,WCCT:309,PSS:439,HMC:1122,LLL:928,SSL:960,DDD:842,B2FS:858,TFB:421}
        g[BoF] = {VCUM:586,WoF:324,JRB:310,CCM:330,AM:300,B1DW:372,WCCT:320,PSS:316,HMC:929,LLL:735,SSL:767,DDD:649,B2FS:665,TFB:310}
        g[WoF] = {VCUM:586,BoF:412,JRB:420,CCM:418,AM:430,B1DW:440,WCCT:432,PSS:456,HMC:1061,LLL:867,SSL:899,DDD:781,B2FS:797,TFB:392}
        g[JRB] = {VCUM:586,BoF:318,WoF:322,CCM:312,AM:320,B1DW:338,WCCT:342,PSS:354,HMC:949,LLL:755,SSL:787,DDD:669,B2FS:685,TFB:294}
        g[CCM] = {VCUM:586,BoF:314,WoF:322,JRB:334,AM:328,B1DW:358,WCCT:322,PSS:340,HMC:959,LLL:765,SSL:797,DDD:679,B2FS:695,TFB:298}
        g[AM] = {VCUM:586,BoF:603,WoF:565,JRB:581,CCM:565,B1DW:563,WCCT:429,PSS:559,HMC:1190,LLL:996,SSL:1028,DDD:910,B2FS:926,TFB:541}
        g[B1DW] = {VCUM:586,BoF:388,WoF:354,JRB:372,CCM:354,AM:382,WCCT:379,PSS:372,HMC:1003,LLL:809,SSL:841,DDD:723,B2FS:739,TFB:384}
        g[WCCT] = {VCUM:453,BoF:228,WoF:190,JRB:206,CCM:190,AM:198,B1DW:188,PSS:184,HMC:867,LLL:673,SSL:705,DDD:587,B2FS:603,TFB:166}
        g[PSS] = {VCUM:453,BoF:228,WoF:190,JRB:206,CCM:190,AM:198,B1DW:188,WCCT:54,HMC:867,LLL:673,SSL:705,DDD:587,B2FS:603,TFB:166,PSS:184}
        g[HMC] = {VCUM:540,BoF:315,WoF:277,JRB:293,CCM:277,AM:285,B1DW:275,WCCT:141,PSS:271,LLL:156,SSL:192,DDD:386,B2FS:402,TFB:253}
        g[LLL] = {VCUM:586,BoF:677,WoF:639,JRB:655,CCM:639,AM:647,B1DW:637,WCCT:503,PSS:633,HMC:300,SSL:166,DDD:516,B2FS:532,TFB:615}
        g[SSL] = {VCUM:586,BoF:681,WoF:643,JRB:659,CCM:643,AM:651,B1DW:641,WCCT:507,PSS:637,HMC:412,LLL:158,DDD:622,B2FS:638,TFB:619}
        g[DDD] = {VCUM:586,BoF:538,WoF:500,JRB:516,CCM:500,AM:508,B1DW:498,WCCT:364,PSS:494,HMC:542,LLL:348,SSL:382,B2FS:118,TFB:476}
        g[B2FS] = {VCUM:586,BoF:490,WoF:452,JRB:468,CCM:452,AM:460,B1DW:450,WCCT:316,PSS:446,HMC:504,LLL:310,SSL:344,DDD:114,TFB:428}
        g[TFB] = {}

    #g.set_visit_count(PSS, 4)

    # goal is 5386
    nodes = [SPAWN, VCUM, WCCT, PSS, PSS, B1DW, WoF, JRB, CCM, BoF, AM, DDD, B2FS, SSL, LLL, HMC, PSS, PSS, TFB] # (/)
    print(sum(g[nodes[i]][nodes[i+1]] for i in range(len(nodes)-1)))
    return

    ##################################################################

    s = time.time()
    route, length = find_routes_with_memo(g, start, end)
    e = time.time()

    print(f"Fastest route from {start} to {end} is [{length}]: {' -> '.join([str(n) for n in route])}")
    print(f"Took {(e-s):.2f}s")


if __name__ == "__main__":
    #main(0)
    main(1)
