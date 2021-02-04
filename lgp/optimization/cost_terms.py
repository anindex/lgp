import numpy as np
from pyrieef.geometry.differentiable_geometry import DifferentiableMap


class LogBarrierFunction(DifferentiableMap):

    """
    Log barrier function

        f(x) = -mu log(x)

    Note fot the sdf the sign has to be flipped, you can set alpha to -1.

    Parameters
    ----------
        g : R^n -> R, constraint function that allways has to be positive
        mu : float
        alpha : float

    """

    def __init__(self, margin=1e-10):
        self.mu = .1
        self._margin = margin

    def output_dimension(self):
        return 1

    def input_dimension(self):
        return 1

    def set_mu(self, mu):
        self._mu = mu

    def forward(self, x):
        """ TODO add this notion of infity """
        # np.Infity throws warnning in current version of linesearch
        # infinity = 1e+200, otherwise does not work
        infinity = np.Infinity
        d = x < self._margin
        if type(x) is float or x.shape == ():
            return infinity if d else -self.mu * np.log(x)
        else:
            return np.where(d, infinity, -self.mu * np.log(x))

    def jacobian(self, x):
        J = np.matrix(np.zeros((1, 1)))
        if x < self._margin:
            return J
        J[0, 0] = -self.mu / x
        return J

    def hessian(self, x):
        H = np.matrix(np.zeros((1, 1)))
        if x < self._margin:
            return H
        H[0, 0] = self.mu / (x ** 2)
        return H
