'''
An HMC sampler for mirt_util.py in conditional_energy_data
'''

from __future__ import division
import numpy as np
import copy as cp

def main():
    '''
    q = np.array([[-1.50,-1.55]])
    p = np.array([[-1,1]])

    q = q.T
    p = p.T
    '''
    q = 0.5
    p = -0.5

    def U(q):
        return q**2 / 2
    def grad_U(q):
        return q

    sample_chain = [q]
    for _ in xrange(500):
        q = HMC(U, grad_U, 0.75, 500, q, 1)
        sample_chain.append(np.asscalar(q))
        #print "sample:", np.asscalar(q)

    mean = sum(sample_chain)/len(sample_chain)
    print "Mean:", mean

    var = 0
    for s in sample_chain:
        var += (s-mean)**2
    var /= len(sample_chain)
    print "variance", var

def HMC(U, grad_U, epsilon, L, current_q, dim):
    q = cp.copy(current_q)
    p = np.random.randn(dim, 1)
    current_p = cp.copy(p)

    p -= epsilon * grad_U(q)/2

    for i in xrange(L):
        #print "q, current_q", q, current_q
        q += epsilon*p
        if i < L-1:
            p -= epsilon*grad_U(q)

    p -= epsilon*grad_U(q)/2
    p = -p

    #print "current_q, q", current_q, q
    #print "current_p, p", current_p, p
    current_U = U(current_q)
    current_K = np.asscalar(np.dot(current_p, current_p.T)) / 2
    proposed_U = U(q)
    proposed_K = np.asscalar(np.dot(p, p.T)) / 2

    #print "current U,K", current_U, current_K
    #print "proposed U,K", proposed_U, proposed_K
    unif_random = np.random.uniform()
    #print "result", np.exp((current_U+current_K) - (proposed_U+proposed_K))
    if np.exp((current_U+current_K) - (proposed_U+proposed_K)) > unif_random:
        return q
    else:
        return current_q

if __name__ == "__main__":
    main()
