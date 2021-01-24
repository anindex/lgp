import logging
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque


class LGPTree(object):
    logger = logging.getLogger(__name__)

    def __init__(self, domain, problem):
        self.domain = domain
        self.problem = problem
        self.tree = nx.Graph(name=self.problem.name)
        self.init_state = self.problem.state
        self.goal_states = []

    def build_tree(self):
        '''
        Build LGP tree from PDDL domain and problem
        '''
        positive_goals = self.problem.positive_goals
        negative_goals = self.problem.negative_goals
        state = self.problem.state
        if LGPTree.applicable(state, positive_goals, negative_goals):
            LGPTree.logger.info('Goals are already achieved! Do nothing.')
            return
        self.tree.clear()
        # Grounding process, i.e. assign parameters substitutions to predicate actions to make propositional actions
        ground_actions = self.domain.ground_actions()
        # BFS Search to build paths
        visited = set([state])
        fringe = deque()
        fringe.append(state)
        while fringe:
            state = fringe.popleft()
            for act in ground_actions:
                if LGPTree.applicable(state, act.positive_preconditions, act.negative_preconditions):
                    new_state = LGPTree.apply(state, act.add_effects, act.del_effects)
                    if new_state not in visited:
                        if LGPTree.applicable(new_state, positive_goals, negative_goals):
                            self.goal_states.append(new_state)  # store goal states
                        self.tree.add_edge(state, new_state, action=act)
                        visited.add(new_state)
                        fringe.append(new_state)

    def plan(self, state=None):
        if self.tree.size() == 0:
            LGPTree.logger.warn('LGP Tree is not built yet! Plan nothing.')
            return []
        if state is None:
            state = self.init_state
        if not self.tree.has_node(state):
            LGPTree.logger.warn('State: %s \n is not recognized in LGP tree! Plan nothing.' % str(state))
            return []
        paths = []
        act_seqs = []
        path = nx.shortest_path(self.tree)
        for g in self.goal_states:
            try:
                p = path[state][g]
                paths.append(p)
                act_seq = [self.tree[p[i]][p[i + 1]]['action'] for i in range(len(p) - 1)]
                act_seqs.append(act_seq)
            except:  # noqa
                LGPTree.logger.warn('No path found between source %s and goal %s' % (str(state), str(g)))
        return paths, act_seqs

    def draw_tree(self, label=True, show=True):
        nx.draw(self.tree, with_labels=label)
        if show:
            plt.show()

    @staticmethod
    def applicable(state, positive, negative):
        return positive.issubset(state) and negative.isdisjoint(state)

    @staticmethod
    def apply(state, positive, negative):
        return state.difference(negative).union(positive)
