import numpy as np

from scipy.special import beta

class MLP_model():
    def __init__(self, d, M, N, T, mu, sigmadiag, Dmu, Vsigma,  f, g, dtype = np.float32):
        self.d = d
        self.M = M
        self.N = N
        self.T = T
        self.mu = mu
        self.sigmadiag = sigmadiag
        self.f = f
        self.g = g
        self.Dmu = Dmu
        self.Vsigma = Vsigma
        self.counts = np.zeros(shape=1, dtype=np.int64)
        self.dtype = dtype
    
    def compute(self, t, x, n):
        def rho_inv(z):
            return np.sqrt(z*(1-z))*beta(0.5, 0.5)

        if n == 0:
            return np.zeros(self.d + 1)
        else:
            a = np.zeros(self.d + 1)
            dt = max((self.T - t), 1e-32)/self.N
            Mn = np.power(self.M, n)
            for i in range(Mn):
                X = x
                DX = np.eye(self.d)
                V = np.zeros(self.d)
                W = np.random.normal(size = [self.N, self.d], scale = np.sqrt(dt)).astype(self.dtype)
                for k in range(self.N):
                    V = V + np.dot(np.transpose(np.dot(self.Vsigma(X), DX)), W[k])
                    DX = DX + self.Dmu(X) * DX * dt
                    drift = self.mu(X) * dt
                    diffu = self.sigmadiag(X) * W[k]
                    X = X + drift + diffu
                    self.counts = self.counts + 4

                V = (1.0 / max((self.T - t), 1e-32)) * V
                vectorV = np.concatenate((np.array([1]), V), axis=None)
                a = a + (self.g(X) - self.g(x)) * vectorV
                self.counts = self.counts + 1

            vector_g = np.array([self.g(x)])

            u = a/Mn + np.concatenate((vector_g, np.zeros(self.d)), axis=None)
            
            for l in range(n):
                Mnl = np.power(self.M, n-l)
                b = np.zeros(self.d + 1)
                for i in range(Mnl):
                    Y = x
                    DX1 = np.eye(self.d)
                    V1 = np.zeros(self.d)
                    R = t + (self.T-t) * np.random.beta(0.5, 0.5)
                    S = np.int32(np.floor((R-t)/dt) + 1)
                    W = np.random.normal(size = [S, self.d], scale = np.sqrt(dt)).astype(self.dtype)
                    self.counts = self.counts + 1 + S * self.d
                    for k in range(S):
                        V1 = V1 + np.dot(np.transpose(np.dot(self.Vsigma(Y), DX1)), W[k])
                        DX1 = DX1 + self.Dmu(Y) * DX1 * dt
                        drift = self.mu(Y) * dt
                        diffu = self.sigmadiag(Y) * W[k]
                        Y = Y + drift + diffu
                        self.counts = self.counts + 4

                    V1 = (1.0 / max((R - t), 1e-32)) * V1
                    vectorV1 = np.concatenate((np.array([1]), V1))
                    b = b + rho_inv((R - t)/max((self.T - t), 1e-32))*self.f(R, Y, self.compute(R, Y, l)[0], self.compute(R, Y, l)[1:]) * vectorV1
                    self.counts = self.counts + 1
                    if l > 0:
                        b = b - rho_inv((R - t)/max((self.T - t), 1e-32))*self.f(R, Y, self.compute(R, Y, l-1)[0], self.compute(R, Y, l-1)[1:]) * vectorV1
                        self.counts = self.counts + 1

                        
                u = u + (self.T-t)*b/Mnl

                
            return u