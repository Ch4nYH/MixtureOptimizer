import numpy as np
from collections import defaultdict

class Optimizer(object):
    def __init__(self, alpha = 0.001, beta1 = 0.9, beta2 = 0.999, eta = 1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        
        self.r = defaultdict(float)
        self.mt = defaultdict(float)
        self.vt = defaultdict(float)
        self.mt_hat = defaultdict(float)
        self.vt_hat = defaultdict(float)
        self.t = 0
        
        self.update_rules = [lambda x, name: self.alpha * x, \
            lambda x, name: self.alpha / (self.eta + np.sqrt(self.r[name])) * x, \
            lambda x, name: self.alpha * self.mt_hat[name] / (np.sqrt(self.vt_hat[name]) + self.eta)]
        
    def update(self, para, grad, name, action):
        # update moments
        self.t = self.t + 1
        self.mt[name] = self.beta1 * self.mt[name] + (1 - self.beta1) * grad
        self.vt[name] = self.beta2 * self.vt[name] + (1 - self.beta2) * (grad ** 2)
        self.mt_hat[name] = self.mt[name]/ (1 - np.power(self.beta1, self.t))
        self.vt_hat[name] = self.vt[name] / (1 - np.power(self.beta2, self.t))
        self.r[name] = self.r[name] + (grad ** 2)
        para.add_(-self.update_rules[action](grad, name))
        
        
    def log(self):
        pass
    def __call__(self, para, grad, names, actions):
        for i in range(para.shape[0]):
            self.update(para[i], grad[i], names[i], actions[i])
