import logging
import re

from lgp.logic.action import Action


class PDDLParser(object):
    '''
    PDDL Parser for reading domain and problem definitions. This supports PDDL 1.2.
    '''
    logger = logging.getLogger(__name__)
    SUPPORTED_REQUIREMENTS = [':strips', ':negative-preconditions', ':typing']

    def __init__(self):
        # init variables
        self.domain_name = 'unknown'
        self.requirements = []
        self.types = {}
        self.objects = {}
        self.actions = []
        self.predicates = {}

    def scan_tokens(self, filename):
        with open(filename, 'r') as f:
            # Remove single line comments
            content = re.sub(r';.*$', '', f.read(), flags=re.MULTILINE).lower()
        # Tokenize
        stack = []
        array = []
        for t in re.findall(r'[()]|[^\s()]+', content):
            if t == '(':
                stack.append(array)
                array = []
            elif t == ')':
                if stack:
                    temp = array
                    array = stack.pop()
                    array.append(temp)
                else:
                    raise Exception('Missing open parentheses')
            else:
                array.append(t)
        if stack:
            raise Exception('Missing close parentheses')
        if len(array) != 1:
            raise Exception('Invalid PDDL expressions! Please check again.')
        return array[0]

    def parse_action(self, group):
        name = group.pop(0)
        if not type(name) is str:
            raise Exception('Action without name definition')
        for act in self.actions:
            if act.name == name:
                raise Exception('Action ' + name + ' redefined')
        parameters = []
        positive_preconditions = []
        negative_preconditions = []
        add_effects = []
        del_effects = []
        extensions = None
        while group:
            t = group.pop(0)
            if t == ':parameters':
                if not type(group) is list:
                    raise Exception('Error with ' + name + ' parameters')
                parameters = []
                untyped_parameters = []
                p = group.pop(0)
                while p:
                    t = p.pop(0)
                    if t == '-':
                        if not untyped_parameters:
                            raise Exception('Unexpected hyphen in ' + name + ' parameters')
                        ptype = p.pop(0)
                        while untyped_parameters:
                            parameters.append([untyped_parameters.pop(0), ptype])
                    else:
                        untyped_parameters.append(t)
                while untyped_parameters:
                    parameters.append([untyped_parameters.pop(0), 'object'])
            elif t == ':precondition':
                self.split_predicates(group.pop(0), positive_preconditions, negative_preconditions, name, ' preconditions')
            elif t == ':effect':
                self.split_predicates(group.pop(0), add_effects, del_effects, name, ' effects')
            else:
                extensions = self.parse_action_extended(t, group)
        self.actions.append(Action(name, parameters, positive_preconditions, negative_preconditions, add_effects, del_effects, extensions))

    def parse_action_extended(self, t, group):
        '''
        This is placeholder function for extensible keywords of actions in PDDL.
        '''
        PDDLParser.logger.warn(str(t) + ' is not recognized in action')

    def parse_predicates(self, group):
        for pred in group:
            predicate_name = pred.pop(0)
            if predicate_name in self.predicates:
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
            self.predicates[predicate_name] = arguments
