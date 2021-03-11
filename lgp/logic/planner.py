import logging
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from operator import itemgetter


class LogicPlanner(object):
    logger = logging.getLogger(__name__)

    def __init__(self, domain, problem, self_edge=False):
        self.domain = domain
        self.problem = problem
        self.self_edge = self_edge
        self.tree = nx.DiGraph(name=self.problem.name)
        self.init_state = self.problem.state
        self.goal_states = set()
        self.build_graph()

    def build_graph(self):
        '''
        Build LGP tree from PDDL domain and problem
        '''
        positive_goals = self.problem.positive_goals
        negative_goals = self.problem.negative_goals
        state = self.problem.state
        self.tree.clear()
        # Grounding process, i.e. assign parameters substitutions to predicate actions to make propositional actions
        ground_actions = self.domain.ground_actions()
        # BFS Search to build paths
        fringe = deque()
        fringe.append(state)
        visited = set()
        while fringe:
            state = fringe.popleft()
            for act in ground_actions:
                if LogicPlanner.applicable(state, act.positive_preconditions, act.negative_preconditions):
                    new_state = LogicPlanner.apply(state, act.add_effects, act.del_effects)
                    if not self.self_edge and new_state == state:  # ignore same state transition
                        continue
                    if new_state not in visited:
                        if LogicPlanner.check_goal(new_state, positive_goals, negative_goals):
                            self.goal_states.add(new_state)  # store goal states
                        self.tree.add_edge(state, new_state, action=act)
                        fringe.append(new_state)
                        visited.add(new_state)

    def plan(self, state=None):
        if self.tree.size() == 0:
            LogicPlanner.logger.warn('LGP Tree is not built yet! Plan nothing.')
            return []
        if state is None:
            state = self.init_state
        if not self.tree.has_node(state):
            LogicPlanner.logger.warn('State: %s \n is not recognized in LGP tree! Plan nothing.' % str(state))
            return []
        paths = []
        act_seqs = []
        path = nx.shortest_path(self.tree, source=state)
        for g in self.goal_states:
            try:
                p = path[g]
                paths.append(p)
                act_seq = [self.tree[p[i]][p[i + 1]]['action'] for i in range(len(p) - 1)]
                act_seqs.append(act_seq)
            except:  # noqa
                LogicPlanner.logger.warn('No path found between source %s and goal %s' % (str(state), str(g)))
        return paths, act_seqs

    def draw_tree(self, init_state=None, paths=None, label=True, show=True):
        node_color = self._color_states(init_state)
        edge_color = None
        if paths is not None:
            edge_color = self._color_edges(paths)
        nx.draw(self.tree, with_labels=label, node_color=node_color, edge_color=edge_color, font_size=5)
        if show:
            plt.show()

    def _color_states(self, init_state=None):
        if init_state is None:
            init_state = self.init_state
        color_map = []
        for n in self.tree:
            if n == init_state:
                color_map.append('green')
            elif n in self.goal_states:
                color_map.append('red')
            else:
                color_map.append('skyblue')
        return color_map

    def _color_edges(self, paths):
        edges = tuple((p[i], p[i + 1]) for p in paths for i in range(len(p) - 1))
        edge_color = []
        for e in self.tree.edges():
            if e in edges:
                edge_color.append('red')
            else:
                edge_color.append('black')
        return edge_color

    @staticmethod
    def applicable(state, positive, negative):
        positive = LogicPlanner.match_any(state, positive)
        negative = LogicPlanner.match_any(state, negative)
        return positive.issubset(state) and negative.isdisjoint(state)

    @staticmethod
    def apply(state, positive, negative):
        # only match any ?* for negative effects
        negative = LogicPlanner.match_any(state, negative)
        return state.difference(negative).union(positive)

    @staticmethod
    def match_any(state, group):
        for p in group:
            if '?*' in p:
                checks = [i for i, v in enumerate(p) if v != '?*']
                for state_p in state:
                    if p[0] in state_p:
                        p_check = ''.join(itemgetter(*checks)(p))
                        state_p_check = ''.join(itemgetter(*checks)(state_p))
                        if p_check == state_p_check:
                            group = group.difference(frozenset([p])).union(frozenset([state_p]))
                            break
        return group
    
    @staticmethod
    def check_goal(state, positive_goals, negative_goals):
        assert len(positive_goals) == len(negative_goals)
        for i in range(len(positive_goals)):
            if LogicPlanner.applicable(state, positive_goals[i], negative_goals[i]):
                return True
        return False
