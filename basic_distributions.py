'''
An HMC sampler for conditional_energy_data in mirt_util.py
'''

from __future__ import division
import numpy as np
import copy as cp
import cPickle as pickle

import matplotlib.pyplot as plt

np.random.seed(555) #for reproducibility

def main():

    #hyperparameters
    num_samples = 1000
    #eps = 0.25 #leapfrog step size

    #initial conditions
    q_init = -0.5

    def U(q):
        return q**2 / 2
    def grad_U(q):
        return q

    lookup_matrix = []
    L_trials = xrange(100, 5001, 100)
    #eps_trials = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    #        1.3, 1.4, 1.5]
    eps_trials = [1.6, 1.7, 1.8, 1.9, 2.0]
    num_L_trials = len(L_trials)
    num_eps_trials = len(eps_trials)
    #populate the lookup matrix
    lookup_matrix = [[(eps,L) for L in L_trials] for eps in
            eps_trials]
    data = [[0 for i in xrange(num_L_trials)] for j in xrange(num_eps_trials)]

    for i in xrange(num_eps_trials):
        print "generating samples for eps=%f ..." % eps_trials[i]
        for j in xrange(num_L_trials):
            eps, L = lookup_matrix[i][j]
            data[i][j] = generate_HMC_samples(U, grad_U, eps, L, q_init, num_samples)

    pickle.dump(data, open("sample_matrix_2.pkl", 'wb'))
    pickle.dump(lookup_matrix, open("lookup_matrix_2.pkl", 'wb'))

def generate_HMC_samples(U, grad_U, eps, L, q_init, num_samples):

    sample_chain = []
    for _ in xrange(num_samples):
        q_new = HMC(U, grad_U, eps, L, q_init, 1)
        sample_chain.append(float(q_new))

    return sample_chain

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

def KS_test(s1, s2):
    pass

if __name__ == "__main__":
    main()
