import logging
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from operator import itemgetter
import os
import sys
import pickle

_path_file = os.path.dirname(os.path.realpath(__file__))


class LogicPlanner(object):
    logger = logging.getLogger(__name__)

    def __init__(self, domain):
        self.domain = domain
        self.cache_path = os.path.join(_path_file, '../../data/caches')
        os.makedirs(self.cache_path, exist_ok=True)
    
    def init_planner(self, **kwargs):
        self.problem = kwargs.get('problem')
        self.self_edge = kwargs.get('self_edge', False)
        self.ignore_cache = kwargs.get('ignore_cache', False)
        self.cache_name = os.path.join(self.cache_path, self.problem.name + '.gpickle')
        # Grounding process, i.e. assign parameters substitutions to predicate actions to make propositional actions
        self.ground_actions = self.domain.ground_actions(self.problem.objects)
        self.current_state = self.problem.state
        self.build_graph()

    def check_cache(self):
        return os.path.isfile(self.cache_name)
    
    def load_cache(self):
        with open(self.cache_name, 'rb') as f:
                data = pickle.load(f)
                self.graph, self.goal_states = data['graph'], data['goals']

    def save_cache(self):
        with open(self.cache_name, 'wb') as f:
            pickle.dump({'graph': self.graph, 'goals': self.goal_states}, f)

    def build_graph(self):
        '''
        Build LGP graph from PDDL domain and problem
        '''
        # check if cache exists
        if not self.ignore_cache and self.check_cache():
            self.load_cache()
            return
        self.graph = nx.DiGraph()
        self.goal_states = set()
        # BFS Search to build paths
        fringe = deque()
        fringe.append(self.problem.state)
        while fringe:
            state = fringe.popleft()
            for act in self.ground_actions:
                if LogicPlanner.applicable(state, act.positive_preconditions, act.negative_preconditions):
                    new_state = LogicPlanner.apply(state, act.add_effects, act.del_effects)
                    if not self.self_edge and new_state == state:  # ignore same state transition
                        continue
                    if not self.graph.has_edge(state, new_state):
                        if LogicPlanner.check_goal(new_state, self.problem.positive_goals, self.problem.negative_goals):
                            self.goal_states.add(new_state)  # store goal states
                        self.graph.add_edge(state, new_state, action=act)
                        fringe.append(new_state)
        self.save_cache()
    
    def resolve_inconsistencies(self, positives, negatives):
        for p in positives:
            if not self.graph.has_edge(p[0], p[1]):
                self.graph.add_edge(p[0],  p[1], action=p[2])
        for p in negatives:
            if self.graph.has_edge(p[0], p[1]):
                self.graph.remove_edge(p[0],  p[1])

    def heuristic(self, s, g):
        '''
        This heuristic is only defined if problem goals are defined.
        '''
        h = 0
        for p in self.problem.positive_goals:
            if p not in s:
                h += 1
        for p in self.problem.negative_goals:
            if p in s:
                h += 1
        return h

    def plan(self, state=None, alternative=False):
        if self.graph.size() == 0:
            LogicPlanner.logger.warn('LGP graph is not built yet! Plan nothing.')
            return [], []
        if state is None:
            state = self.current_state
        if not self.graph.has_node(state):
            # check if current state could connected to feasibility graph
            for act in self.ground_actions:
                if LogicPlanner.applicable(state, act.positive_preconditions, act.negative_preconditions):
                    new_state = LogicPlanner.apply(state, act.add_effects, act.del_effects)
                    if new_state == state:  # ignore same state transition
                        continue
                    self.graph.add_edge(state, new_state, action=act)
            if not self.graph.has_node(state):
                LogicPlanner.logger.warn('State: %s \n is not recognized in LGP graph. Could not find feasible path from this state to goal!.' % str(state))
                return [], []
        paths = []
        act_seqs = []
        for goal in self.goal_states:
            try:
                if alternative:
                    all_paths = [path for path in nx.all_shortest_paths(self.graph, source=state, target=goal)]
                    paths.extend(all_paths)
                    for path in all_paths:
                        act_seq = [self.graph[path[i]][path[i + 1]]['action'] for i in range(len(path) - 1)]
                        act_seqs.append(act_seq)
                else:
                    path = nx.astar_path(self.graph, source=state, target=goal, heuristic=self.heuristic)
                    paths.append(path)
                    act_seq = [self.graph[path[i]][path[i + 1]]['action'] for i in range(len(path) - 1)]
                    act_seqs.append(act_seq)
            except:
                LogicPlanner.logger.warn(f'{goal} is not reachable from {state}.')
        return paths, act_seqs

    def draw_tree(self, current_state=None, paths=None, label=True, show=True):
        node_color = self._color_states(current_state)
        edge_color = None
        if paths is not None:
            edge_color = self._color_edges(paths)
        nx.draw(self.graph, with_labels=label, node_color=node_color, edge_color=edge_color, font_size=5)
        if show:
            plt.show()

    def _color_states(self, current_state=None):
        if current_state is None:
            current_state = self.current_state
        color_map = []
        for n in self.graph:
            if n == current_state:
                color_map.append('green')
            elif n in self.goal_states:
                color_map.append('red')
            else:
                color_map.append('skyblue')
        return color_map

    def _color_edges(self, paths):
        edges = tuple((p[i], p[i + 1]) for p in paths for i in range(len(p) - 1))
        edge_color = []
        for e in self.graph.edges():
            if e in edges:
                edge_color.append('red')
            else:
                edge_color.append('black')
        return edge_color

    @staticmethod
    def applicable(state, positive, negative):
        # positive = LogicPlanner.match_any(state, positive)  # uncomment if ?* operator is in preconditions
        # negative = LogicPlanner.match_any(state, negative)
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
