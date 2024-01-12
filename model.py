import numpy as np
import os
import time
from MultiLevelPicard import MLP_model

path = os.path.join(os.getcwd(), "test_mlp/")
dtype = np.float32

N = 12
Mnmax = 5
T = 0.25

mu0 = 0.06
sigma = 0.2
L = 10
K0 = 25
K1 = 95
K2 = 120
K = 50


def g(x):
    return max(0.0, np.max(x)-K1) - 2*max(0.0, np.max(x)-K2)

dd = [10,100,200,300]
runs = 10

print("======================================================================")
for i in range(len(dd)):
    d = dd[i]

    def f(t, x, y, z):
        return (L/d) * max(0, max(np.max(z), np.max(-z)) - K0)

    def mu(x):
        return mu0*x
    
    def sigmadiag(x):
        return sigma

    def Dmu(x):
        return mu0

    def Vsigma(x):
        return (sigma ** -1)*np.eye(d)
    
    sol = np.zeros([Mnmax, runs])
    mgrad = np.zeros([Mnmax, runs])
    tms = np.zeros([Mnmax, runs])
    fev = np.zeros([Mnmax, runs])
    for M in range(1, Mnmax+1):
        for r in range(runs):
            b = time.time()
            mlp = MLP_model(d, M, N, T, mu, sigmadiag, Dmu, Vsigma, f, g)
            x0 = 100.0*np.ones(d, dtype = dtype)
            sol[M-1, r] = mlp.compute(0.0, x0, M)[0]
            mgrad[M - 1, r] = max(np.max(mlp.compute(0.0, x0, M)[1:]), np.max((-1)*mlp.compute(0.0, x0, M)[1:]))
            e = time.time()
            tms[M-1, r] = e-b
            fev[M - 1, r] = mlp.counts
            print("MLP performed for d = " + str(d) + ", m = " + str(M) + "/" + str(Mnmax) + ", run " + str(r+1) + "/" + str(runs) + ", in " + str(np.round(tms[M-1, r], 1)) + "s with solution " + str(sol[M-1, r]) + " and maxgrad " + str(mgrad[M-1, r]))
            
        np.savetxt(path + "mlp_sol_" + str(dd[i]) + ".csv", sol)
        np.savetxt(path + "mlp_tms_" + str(dd[i]) + ".csv", tms)
        np.savetxt(path + "mlp_fev_" + str(dd[i]) + ".csv", fev)

    
print("======================================================================")
print("MLP solutions saved")