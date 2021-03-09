import logging
import re

from lgp.logic.action import Action
from lgp.logic.domain import Domain
from lgp.logic.problem import Problem
from lgp.utils.helpers import frozenset_of_tuples


class PDDLParser(object):
    '''
    PDDL Parser for reading domain and problem definitions. This supports PDDL 1.2.
    '''
    logger = logging.getLogger(__name__)
    SUPPORTED_REQUIREMENTS = [':strips', ':negative-preconditions', ':typing']

    @staticmethod
    def scan_tokens(filename):
        with open(filename, 'r') as f:
            # Remove single line comments
            content = re.sub(r';.*$', '', f.read(), flags=re.MULTILINE).lower()
        # Tokenize
        stack = []
        queue = []
        for t in re.findall(r'[()]|[^\s()]+', content):
            if t == '(':
                stack.append(queue)
                queue = []
            elif t == ')':
                if stack:
                    temp = queue
                    queue = stack.pop()
                    queue.append(temp)
                else:
                    raise Exception('Missing open parentheses')
            else:
                queue.append(t)
        if stack:
            raise Exception('Missing close parentheses')
        if len(queue) != 1:
            raise Exception('Invalid PDDL expressions! Please check again.')
        return queue[0]

    @staticmethod
    def parse_domain(domain_filename):
        tokens = PDDLParser.scan_tokens(domain_filename)
        if tokens.pop(0) == 'define':
            domain = Domain()
            for group in tokens:
                t = group.pop(0)  # remove tags
                if t == 'domain':
                    domain.name = group[0]
                elif t == ':requirements':
                    for req in group:
                        if req not in PDDLParser.SUPPORTED_REQUIREMENTS:
                            raise Exception('Requirement ' + req + ' not supported')
                    domain.requirements = group
                elif t == ':constants':
                    domain.constants = PDDLParser.parse_hierarchy(group, t)
                elif t == ':types':
                    domain.types = PDDLParser.parse_hierarchy(group, t)
                elif t == ':predicates':
                    domain.predicates = PDDLParser.parse_predicates(group)
                elif t == ':action':
                    act = PDDLParser.parse_action(group)
                    domain.actions[act.name] = act
                else:
                    domain.extensions = PDDLParser.parse_domain_extended(group, t)
        else:
            raise Exception('File ' + domain_filename + ' does not match PDDL domain syntax!')
        return domain

    @staticmethod
    def parse_domain_extended(group, t):
        PDDLParser.logger.warn(str(t) + ' is not recognized in domain')

    @staticmethod
    def parse_problem(problem_filename):
        tokens = PDDLParser.scan_tokens(problem_filename)
        if tokens.pop(0) == 'define':
            problem = Problem()
            for group in tokens:
                t = group.pop(0)
                if t == 'problem':
                    problem.name = group[0]
                elif t == ':domain':
                    problem.domain_name = group[0]
                elif t == ':requirements':
                    pass  # Ignore requirements in problem, parse them in the domain
                elif t == ':objects':
                    problem.objects = PDDLParser.parse_hierarchy(group, t)
                elif t == ':init':
                    problem.state = frozenset_of_tuples(group)
                elif t == ':goal':
                    positive_goals, negative_goals = PDDLParser.split_predicates(group[0], '', 'goals')
                    problem.positive_goals = frozenset_of_tuples(positive_goals)
                    problem.negative_goals = frozenset_of_tuples(negative_goals)
                else:
                    problem.extensions = PDDLParser.parse_problem_extended(group, t)
        else:
            raise Exception('File ' + problem_filename + ' does not match problem pattern')
        return problem

    @staticmethod
    def parse_problem_extended(group, t):
        PDDLParser.logger.warn(str(t) + ' is not recognized in problem')

    @staticmethod
    def parse_hierarchy(group, name, redefine=False):
        queue = []
        structure = {}
        while group:
            if not redefine and group[0] in structure:
                raise Exception('Redefined supertype of ' + group[0])
            elif group[0] == '-':
                if not queue:
                    raise Exception('Unexpected hyphen in ' + name)
                group.pop(0)
                typ = group.pop(0)
                if typ not in structure:
                    structure[typ] = []
                structure[typ] += queue
                queue = []
            else:
                queue.append(group.pop(0))
        if queue:
            if 'object' not in structure:
                structure['object'] = []
            structure['object'] += queue
        return structure

    @staticmethod
    def parse_action(group):
        name = group.pop(0)
        if type(name) is not str:
            raise Exception('Action without name definition')
        action = Action(name=name)
        while group:
            t = group.pop(0)
            if t == ':parameters':
                action.parameters = PDDLParser.parse_action_parameters(group.pop(0), name)
            elif t == ':precondition':
                positive_preconditions, negative_preconditions = PDDLParser.split_predicates(group.pop(0), name, ' preconditions')
                action.positive_preconditions, action.negative_preconditions = frozenset_of_tuples(positive_preconditions), frozenset_of_tuples(negative_preconditions)
            elif t == ':effect':
                add_effects, del_effects = PDDLParser.split_predicates(group.pop(0), name, ' effects')
                action.add_effects, action.del_effects = frozenset_of_tuples(add_effects), frozenset_of_tuples(del_effects)
            else:
                action.extensions[t] = PDDLParser.parse_action_extended(group.pop(0), t)
        return action

    @staticmethod
    def parse_action_parameters(group, name):
        parameters = []
        untyped_parameters = []
        while group:
            t = group.pop(0)
            if t == '-':
                if not untyped_parameters:
                    raise Exception('Unexpected hyphen in ' + name + ' parameters')
                ptype = group.pop(0)
                while untyped_parameters:
                    parameters.append([untyped_parameters.pop(0), ptype])
            else:
                untyped_parameters.append(t)
        while untyped_parameters:
            parameters.append([untyped_parameters.pop(0), 'object'])
        return parameters

    @staticmethod
    def parse_action_extended(group, t):
        '''
        This is placeholder function for extensible keywords of actions in PDDL.
        '''
        if t == ':undo':
            group.pop(0)
            undo_action = PDDLParser.parse_action(group)
            return undo_action
        return None

    @staticmethod
    def parse_predicates(group):
        predicates = {}
        for pred in group:
            predicate_name = pred.pop(0)
            if predicate_name in predicates:
                raise Exception('Predicate ' + predicate_name + ' redefined')
            arguments = {}
            untyped_variables = []
            while pred:
                t = pred.pop(0)
                if t == '-':
                    if not untyped_variables:
                        raise Exception('Unexpected hyphen in predicates')
                    typ = pred.pop(0)
                    while untyped_variables:
                        arguments[untyped_variables.pop(0)] = typ
                else:
                    untyped_variables.append(t)
            while untyped_variables:
                arguments[untyped_variables.pop(0)] = 'object'
            predicates[predicate_name] = arguments
        return predicates

    @staticmethod
    def split_predicates(group, name, part):
        if not group:
            return [], []
        negative = []
        positive = []
        if group[0] == 'and':
            group.pop(0)
        else:
            group = [group]
        for predicate in group:
            if predicate[0] == 'not':
                if len(predicate) != 2:
                    raise Exception('Unexpected not in ' + name + part)
                negative.append(predicate[-1])
            else:
                positive.append(predicate)
        return positive, negative
