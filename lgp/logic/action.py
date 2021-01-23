import itertools

from lgp.utils.helpers import frozenset_of_tuples


class Action:
    '''
    An Action schema with +/- preconditions and +/- effects.
    This class follows PDDL action schema.
    '''
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'unknown')
        self.parameters = kwargs.get('parameters', [])
        self.positive_preconditions = frozenset_of_tuples(kwargs.get('positive_preconditions', []))
        self.negative_preconditions = frozenset_of_tuples(kwargs.get('negative_preconditions', []))
        self.add_effects = frozenset_of_tuples(kwargs.get('add_effects', []))
        self.del_effects = frozenset_of_tuples(kwargs.get('del_effects', []))

    def groundify(self, constants, types):
        '''
        Ground actions with constants (propositional actions)
        '''
        if not self.parameters:
            yield self
            return
        type_map = []
        variables = []
        for var, typ in self.parameters:
            type_stack = [typ]
            items = []
            while type_stack:
                t = type_stack.pop()
                if t in constants:
                    items += constants[t]
                elif t in types:
                    type_stack += types[t]
                else:
                    raise Exception('Unrecognized type ' + t)
            type_map.append(items)
            variables.append(var)
        for assignment in itertools.product(*type_map):
            positive_preconditions = self.replace(self.positive_preconditions, variables, assignment)
            negative_preconditions = self.replace(self.negative_preconditions, variables, assignment)
            add_effects = self.replace(self.add_effects, variables, assignment)
            del_effects = self.replace(self.del_effects, variables, assignment)
            yield Action(name=self.name, parameters=assignment,
                         positive_preconditions=positive_preconditions, negative_preconditions=negative_preconditions,
                         add_effects=add_effects, del_effects=del_effects)

    def replace(self, group, variables, assignment):
        g = []
        for pred in group:
            pred = list(pred)
            iv = 0
            for v in variables:
                while v in pred:
                    pred[pred.index(v)] = assignment[iv]
                iv += 1
            g.append(pred)
        return g

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return 'action: ' + self.name + \
               '\n  parameters: ' + str(self.parameters) + \
               '\n  positive_preconditions: ' + str([list(i) for i in self.positive_preconditions]) + \
               '\n  negative_preconditions: ' + str([list(i) for i in self.negative_preconditions]) + \
               '\n  add_effects: ' + str([list(i) for i in self.add_effects]) + \
               '\n  del_effects: ' + str([list(i) for i in self.del_effects]) + '\n'
