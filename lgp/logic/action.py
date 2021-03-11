import itertools

from lgp.utils.helpers import frozenset_of_tuples


class Action:
    '''
    An Action schema with +/- preconditions and +/- effects.
    This class follows PDDL 1.2 action schema.
    '''
    UNDO_TAG = ':undo'

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'unknown')
        self.parameters = kwargs.get('parameters', [])
        self.positive_preconditions = frozenset_of_tuples(kwargs.get('positive_preconditions', []))
        self.negative_preconditions = frozenset_of_tuples(kwargs.get('negative_preconditions', []))
        self.add_effects = frozenset_of_tuples(kwargs.get('add_effects', []))
        self.del_effects = frozenset_of_tuples(kwargs.get('del_effects', []))
        self.extensions = kwargs.get('extensions', {})

    def get_variables(self):
        return [var for var, _ in self.parameters]

    def groundify(self, constants, types):
        '''
        Ground actions with constants (propositional actions)
        '''
        if not self.parameters:
            yield self
            return
        type_map = []
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
        variables = self.get_variables()
        undo = self.extensions[Action.UNDO_TAG] if Action.UNDO_TAG in self.extensions else None
        if undo is not None:
            undo_variables = undo.get_variables()
        for assignment in itertools.product(*type_map):
            assignment_map = dict(zip(variables, assignment))
            positive_preconditions = Action.replace(self.positive_preconditions, assignment_map)
            negative_preconditions = Action.replace(self.negative_preconditions, assignment_map)
            add_effects = Action.replace(self.add_effects, assignment_map)
            del_effects = Action.replace(self.del_effects, assignment_map)
            # ground undo action
            grounded_undo = None
            if undo is not None:
                undo_assigment = [assignment_map[v] for v in undo_variables]
                undo_assigment_map = dict(zip(undo_variables, undo_assigment))
                undo_pospre = Action.replace(undo.positive_preconditions, undo_assigment_map)
                undo_negpre = Action.replace(undo.negative_preconditions, undo_assigment_map)
                undo_addeffects = Action.replace(undo.add_effects, undo_assigment_map)
                undo_deleffects = Action.replace(undo.del_effects, undo_assigment_map)
                grounded_undo = Action(name=undo.name, parameters=undo_assigment,
                                       positive_preconditions=undo_pospre, negative_preconditions=undo_negpre,
                                       add_effects=undo_addeffects, del_effects=undo_deleffects)
            yield Action(name=self.name, parameters=assignment,
                         positive_preconditions=positive_preconditions, negative_preconditions=negative_preconditions,
                         add_effects=add_effects, del_effects=del_effects, extensions={Action.UNDO_TAG: grounded_undo})

    @staticmethod
    def replace(group, assignment_map):
        g = []
        for pred in group:
            pred = list(pred)
            for i, token in enumerate(pred):
                if token in assignment_map:
                    pred[i] = assignment_map[token]
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


class DurativeAction(Action):
    '''
    A Durative Action schema with duration, at start & end with +/- preconditions and +/- effects.
    This class follows PDDL 2.1 action schema.
    '''
    def __init__(self, **kwargs):
        super(DurativeAction, self).__init__(**kwargs)
        self.duration = kwargs.get('duration', None) # could be a function or a number
        self.start_positive_preconditions = frozenset_of_tuples(kwargs.get('start_positive_preconditions', []))
        self.start_negative_preconditions = frozenset_of_tuples(kwargs.get('start_negative_preconditions', []))
        self.end_positive_preconditions = frozenset_of_tuples(kwargs.get('end_positive_preconditions', []))
        self.end_negative_preconditions = frozenset_of_tuples(kwargs.get('end_negative_preconditions', []))
        self.start_add_effects = frozenset_of_tuples(kwargs.get('start_add_effects', []))
        self.start_del_effects = frozenset_of_tuples(kwargs.get('start_del_effects', []))
        self.end_add_effects = frozenset_of_tuples(kwargs.get('end_add_effects', []))
        self.end_del_effects = frozenset_of_tuples(kwargs.get('end_del_effects', []))

    def groundify(self, constants, types):
        '''
        Ground durative actions with constants (propositional actions)
        '''
        if not self.parameters:
            yield self
            return
        type_map = []
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
        variables = self.get_variables()
        for assignment in itertools.product(*type_map):
            assignment_map = dict(zip(variables, assignment))
            start_positive_preconditions = Action.replace(self.start_positive_preconditions, assignment_map)
            start_negative_preconditions = Action.replace(self.start_negative_preconditions, assignment_map)
            end_positive_preconditions = Action.replace(self.end_positive_preconditions, assignment_map)
            end_negative_preconditions = Action.replace(self.end_negative_preconditions, assignment_map)
            positive_preconditions = start_positive_preconditions + end_positive_preconditions
            negative_preconditions = start_negative_preconditions + end_negative_preconditions
            start_add_effects = Action.replace(self.start_add_effects, assignment_map)
            start_del_effects = Action.replace(self.start_del_effects, assignment_map)
            end_add_effects = Action.replace(self.end_add_effects, assignment_map)
            end_del_effects = Action.replace(self.end_del_effects, assignment_map)
            add_effects = start_add_effects + end_add_effects
            del_effects = start_del_effects + end_del_effects
            yield DurativeAction(name=self.name, parameters=assignment,
                                 positive_preconditions=positive_preconditions, negative_preconditions=negative_preconditions,
                                 start_positive_preconditions=start_positive_preconditions, start_negative_preconditions=start_negative_preconditions,
                                 end_positive_preconditions=end_positive_preconditions, end_negative_preconditions=end_negative_preconditions,
                                 add_effects=add_effects, del_effects=del_effects,
                                 start_add_effects=start_add_effects, start_del_effects=start_del_effects,
                                 end_add_effects=end_add_effects, end_del_effects=end_del_effects)
    
    def __str__(self):
        return 'durative-action: ' + self.name + \
               '\n  parameters: ' + str(self.parameters) + \
               '\n  start_positive_preconditions: ' + str([list(i) for i in self.start_positive_preconditions]) + \
               '\n  start_negative_preconditions: ' + str([list(i) for i in self.start_negative_preconditions]) + \
               '\n  end_positive_preconditions: ' + str([list(i) for i in self.end_positive_preconditions]) + \
               '\n  end_negative_preconditions: ' + str([list(i) for i in self.end_negative_preconditions]) + \
               '\n  start_add_effects: ' + str([list(i) for i in self.start_add_effects]) + \
               '\n  start_del_effects: ' + str([list(i) for i in self.start_del_effects]) + \
               '\n  end_add_effects: ' + str([list(i) for i in self.end_add_effects]) + \
               '\n  end_del_effects: ' + str([list(i) for i in self.end_del_effects]) + '\n'
    

        
