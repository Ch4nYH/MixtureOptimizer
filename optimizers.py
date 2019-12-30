import numpy as np

class Optimizer(object):
    def __init__(self):
        pass
    def update(self, para, grad):
        pass
    def log(self):
        pass

    def __call__(self, para, grad):
        for i in range(para.shape[0]):
            self.update(para[i], grad[i].item())

class Adam(Optimizer):

    def __init__(self, alpha = 0.001, beta1 = 0.9, beta2 = 0.999, eta = 1e-8):
        super(Adam, self).__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta

        self.t = 0
        
        self.mt = 0
        self.vt = 0
        self.mt_hat = 0
        self.vt_hat = 0 

    def update(self, para, grad):
        self.t = self.t + 1
        self.mt = self.beta1 * self.mt + (1 - self.beta1) * grad
        self.vt = self.beta2 * self.vt + (1 - self.beta2) * (grad ** 2)
        self.mt_hat = self.mt / (1 - np.power(self.beta1, self.t))
        self.vt_hat = self.vt / (1 - np.power(self.beta2, self.t))
        para.add_(-self.alpha * self.mt_hat / (np.sqrt(self.vt_hat) + self.eta))

class AdaGrad(Optimizer):

    def __init__(self, alpha = 0.001, eta = 1e-8):
        super(AdaGrad, self).__init__()
        self.alpha = alpha
        self.eta = eta

        self.r = 0

    def update(self, para, grad):
        self.r = self.r + (grad ** 2)
        para.add_(-self.alpha / (self.eta + np.sqrt(self.r)) * grad)

class SGD(Optimizer):

    def __init__(self, alpha = 0.001):
        super(SGD, self).__init__()
        self.alpha = alpha

    def update(self, para, grad):
        para.add_(-self.alpha * grad)