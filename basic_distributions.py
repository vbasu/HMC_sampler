'''
An HMC sampler for conditional_energy_data in mirt_util.py
'''

from __future__ import division
import numpy as np
import copy as cp

def main():

    #hyperparameters
    num_samples = 500
    eps = 0.75 #leapfrog step size
    L = 1000 #leapfrog steps per iteration

    #initial conditions
    q_init = 0.5
    #p = -0.5

    def U(q):
        return q**2 / 2
    def grad_U(q):
        return q

    sample_chain = []
    for _ in xrange(num_samples):
        q_new = HMC(U, grad_U, eps, L, q_init, 1)
        sample_chain.append(float(q_new))

    print "HMC mean:", mean(sample_chain)
    print "HMC variance:", variance(sample_chain)

    ideal_chain = np.random.normal(0, 1, num_samples)

    print "mean:", mean(ideal_chain)
    print "variance:", variance(ideal_chain)

def HMC(U, grad_U, epsilon, L, current_q, dim):
    q = cp.copy(current_q)
    p = np.random.randn(dim, 1)
    current_p = cp.copy(p)

    p -= epsilon * grad_U(q)/2

    for i in xrange(L):
        q += epsilon*p
        if i < L-1:
            p -= epsilon*grad_U(q)

    p -= epsilon*grad_U(q)/2
    p = -p

    current_U = U(current_q)
    current_K = np.asscalar(np.dot(current_p, current_p.T)) / 2
    proposed_U = U(q)
    proposed_K = np.asscalar(np.dot(p, p.T)) / 2

    unif_random = np.random.uniform()
    if np.exp((current_U+current_K) - (proposed_U+proposed_K)) > unif_random:
        return q
    else:
        return current_q

def mean(samples):
    if len(samples) == 0: return 0
    return sum(samples)/len(samples)

def variance(samples):
    if len(samples) == 0: return 0
    m = mean(samples)
    var = 0
    for s in samples:
        var += (s-m)**2
    var /= len(samples)
    return var


if __name__ == "__main__":
    main()
