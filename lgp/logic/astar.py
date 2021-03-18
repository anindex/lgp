import logging
from heapq import heappush, heappop
from itertools import count

from networkx.algorithms.shortest_paths.weighted import _weight_function


logger = logging.getLogger(__name__)


def _get_paths(paths, tracks, path, n):
    '''
    Get all possible paths in track dictionary.
    This is a recursive algorithm.
    '''
    if tracks[n] is None or tracks[n][0] is None:
        res = path[:]
        res.reverse()
        paths.append(res)
        return
    for next_n in tracks[n]:
        path.append(next_n)
        _get_paths(paths, tracks, path, next_n)
        path.pop()


def astar(G, source, target, heuristic=None, weight="weight"):
    """
    Returns a list of all possible shortest paths (no breaking tie)
    """
    if source not in G or target not in G:
        logger.error(f'{source} or {target} is not in graph!')
        raise ValueError()
    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0
    weight = _weight_function(G, weight)
    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.
    c = count()
    queue = [(0, next(c), source, 0, None)]
    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}
    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = heappop(queue)
        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue
            # Skip bad paths that were enqueued before finding a better one
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue
        if curnode not in explored:
            explored[curnode] = []
        explored[curnode].append(parent)
        if curnode == target:
            paths = []
            path = [target]
            _get_paths(paths, explored, path, target)
            return paths
        for neighbor, w in G[curnode].items():
            ncost = dist + weight(curnode, neighbor, w)
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                # if qcost < ncost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if qcost < ncost:
                    continue
            else:
                h = heuristic(neighbor, target)
            enqueued[neighbor] = ncost, h
            heappush(queue, (ncost + h, next(c), neighbor, ncost, curnode))
    # goal not reached
    raise ValueError(f"Node {target} not reachable from {source}")


if __name__ == '__main__':
    tracks = {6: [5, 4], 5: [3], 4: [3], 3: [2, 1], 2: [0], 1: [0], 0: None}
    paths = []
    _get_paths(paths, tracks, [6], 6)
    print(paths)