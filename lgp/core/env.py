from lgp.core.solver import LGP


class Environment(object):
    def __init__(self, **kwargs):
        domain_file = kwargs.get('domain_file')
        problem_file = kwargs.get('problem_file')
        config_file = kwargs.get('config_file')
        self.lgp = LGP(domain_file, problem_file, config_file)
