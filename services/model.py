import math
from scipy.optimize import fsolve
from collections.abc import Iterable


class EconomicModel(object):
    def __init__(self):
        self.set_default()
        
    def set_default(self):
        self.alpha = 0.5
        self.beta = 1.5
        self.gamma = 1.5
        self.delta = 0.1
        self.nu = 5
        self.mu = 20
        self.lambda_ = 20
        self.rho = 10
        self.A0 = 1
        self.L0 = 1
        self.D0 = 1
        self.tau = 0.6
        self.sigma = 0.5
        self.theta = (1 + self.alpha * (self.beta - 1)) ** (-1)

    def L1(self, x):
        return x[3] * ((1 - self.alpha) * self.A0 * x[1] / x[2]) ** (1 / self.alpha)

    def Q1(self, x):
        return self.A0 * x[3] ** self.alpha * self.L1(x) ** (1 - self.alpha)

    def D1(self, x):
        return self.D0 * math.exp(-self.beta * x[1]) * x[5] / (x[1] + x[5])

    def S1(self, x):
        return self.L0 * (1 - math.exp(-self.gamma * x[2])) * x[2] / (x[2] + x[6])

    def I1(self, x):
        return (1 - self.tau) * (1 - self.theta) * x[0]

    def L2(self, x):
        return x[7] * ((1 - self.alpha) * self.A0 * x[5] / x[6]) ** (1 / self.alpha)

    def Q2(self, x):
        return self.A0 * x[7] ** self.alpha * self.L2(x) ** (1 - self.alpha)

    def D2(self, x):
        return self.D0 * math.exp(-self.beta * x[5]) * x[1] / (x[1] + x[5])

    def S2(self, x):
        return self.L0 * (1 - math.exp(-self.gamma * x[6])) * x[6] / (x[2] + x[6])

    def I2(self, x):
        return (1 - self.theta) * x[4]

    def T(self, x):
        return self.tau * x[0]

    def G(self, x):
        """Доход государства"""
        return (1 - self.sigma) * self.tau * x[0]
    
    def G1(self, x):
        """Доход легального сектора"""
        return (1 - self.tau) * self.theta * x[0]
    
    def G2(self, x):
        """Доход теневого сектора"""
        return self.theta * x[4]
    
    def count_model(self, x):
        P1 = (x[1] * min(self.Q1(x), self.D1(x))\
            - x[2] * min(self.L1(x), self.S1(x)) - x[0]) / self.nu
        
        p1 = (self.D1(x) - self.Q1(x)) / self.mu
        
        w1 = (self.L1(x) - self.S1(x)) / self.lambda_
        
        K1 = -self.delta * x[3] + self.I1(x)
        
        P2 = (math.exp(-self.rho * self.sigma * self.T(x))\
            * x[5]\
            * min(self.Q2(x), self.D2(x)) - x[6]\
            * min(self.L2(x), self.S2(x)) - x[4]) / self.nu
        
        p2 = (self.D2(x) - self.Q2(x)) / self.mu
        
        w2 = (self.L2(x) - self.S2(x)) / self.lambda_
        
        K2 = -self.delta * x[7] + self.I2(x)

        result = [P1, p1, w1, K1, P2, p2, w2, K2]
        return result
        
    def __call__(self, x, changed_params={}):
        for param_name in changed_params:
            if not hasattr(self, param_name):
                raise ValueError(param_name)
            setattr(self, param_name, changed_params[param_name])
            
        result = self.count_model(x)
        self.set_default()
        return result


def G_by_x(model: EconomicModel, tau: float, init: list[float] | None = None) -> float:
    if isinstance(tau, Iterable):
        return list(map(lambda x: G_by_x(model, x), list(tau)))
    
    prepared_model_call = lambda x, *args: model(x, changed_params={"tau": args[0]})
    
    # начальные значения модели
    if init is None:
        init = [0, 0.5, 0.25, 0.1, 0, 0.5, 0.25, 0.1]
    new_x = fsolve(prepared_model_call, x0=init, args=(tau,))
    
    model.tau = tau
    result = model.G(new_x)
    
    # сброс до базовых значений модели
    model.set_default()
    
    return result