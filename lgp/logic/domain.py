import logging


class Domain(object):
    '''
    A PDDL domain schema
    '''
    logger = logging.getLogger(__name__)

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'unknown')
        self.requirements = kwargs.get('requirements', [])
        self.types = kwargs.get('types', {})
        self.constants = kwargs.get('constants', {})
        self.predicates = kwargs.get('predicates', {})
        self.functions = kwargs.get('functions', {})
        self.actions = kwargs.get('actions', {})
        self.extensions = kwargs.get('extensions', {})

    def ground_actions(self, objects={}):
        if not objects:
            objects = self.constants
        grounded_actions = []
        for action in self.actions.values():
            for act in action.groundify(objects, self.types):
                grounded_actions.append(act)
        return grounded_actions

    def __str__(self):
        return 'Domain name: ' + self.name + \
               '\nRequirements: ' + str(self.requirements) + \
               '\nTypes: ' + str(self.types) + \
               '\nConstants: ' + str(self.constants) + \
               '\nPredicates: ' + str(self.predicates) + \
               '\nFunctions: ' + str(self.functions) + \
               '\nActions: ' + str([str(a) for a in self.actions]) + '\n'
