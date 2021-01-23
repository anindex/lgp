import logging
from collections import deque


class PDDLPlanner(object):
    logger = logging.getLogger(__name__)

    @staticmethod
    def plan(domain, problem):
        '''
        This implementation does not require tree data structure
        TODO: Maybe tree structure is more efficient?
        '''
        positive_goals = problem.positive_goals
        negative_goals = problem.negative_goals
        state = problem.state
        if PDDLPlanner.applicable(state, positive_goals, negative_goals):
            PDDLPlanner.logger.info('Goals are already achieved! Do nothing.')
            return []
        # Grounding process, i.e. assign parameters substitutions to predicate actions to make propositional actions
        ground_actions = domain.ground_actions()
        # BFS Search
        visited = set([state])
        fringe = deque()
        fringe.extend([state, None])
        while fringe:
            state = fringe.popleft()
            plan = fringe.popleft()
            for act in ground_actions:
                if PDDLPlanner.applicable(state, act.positive_preconditions, act.negative_preconditions):
                    new_state = PDDLPlanner.apply(state, act.add_effects, act.del_effects)
                    if new_state not in visited:
                        if PDDLPlanner.applicable(new_state, positive_goals, negative_goals):
                            full_plan = [act]
                            while plan:
                                act, plan = plan
                                full_plan.insert(0, act)
                            return full_plan
                        visited.add(new_state)
                        fringe.append(new_state)
                        fringe.append((act, plan))
        return None

    @staticmethod
    def applicable(state, positive, negative):
        return positive.issubset(state) and negative.isdisjoint(state)

    @staticmethod
    def apply(state, positive, negative):
        return state.difference(negative).union(positive)
