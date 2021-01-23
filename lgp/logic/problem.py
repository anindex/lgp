class Problem(object):
    '''
    A PDDL problem schema
    '''
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'unknown')
        self.domain_name = kwargs.get('domain_name', 'unknown')
        self.objects = kwargs.get('objects', {})
        self.state = kwargs.get('state', frozenset())
        self.positive_goals = kwargs.get('positive_goals', frozenset())
        self.negative_goals = kwargs.get('negative_goals', frozenset())
        self.extensions = None

    def __str__(self):
        return 'Problem name: ' + self.name + \
               '\nDomain name: ' + self.domain_name + \
               '\nObjects: ' + str(self.objects) + \
               '\nInit state: ' + str([list(i) for i in self.state]) + \
               '\nPositive goals: ' + str([list(i) for i in self.positive_goals]) + \
               '\nNegative goals: ' + str([list(i) for i in self.negative_goals]) + '\n'
