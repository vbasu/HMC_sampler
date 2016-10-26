'''
Generate some plots of the HMC one-step data
'''

from __future__ import division
import numpy as np
import cPickle as pickle

import matplotlib.pyplot as plt

np.random.seed(5) #for reproducibility

def main():
    data = pickle.load(open("sample_matrix.pkl", 'rb'))
    lookup_matrix = pickle.load(open("lookup_matrix.pkl", 'rb'))

    num_samples = 1000

    true_samples = np.random.normal(0,1,num_samples)

    num_L_trials = len(data[0])
    num_eps_trials = len(data)
    #row = 3 #eps
    #eps = (row+16)/10

    for row in xrange(15):
        eps = (row+1)/10
        L_trials = range(100, 5001, 100)
        ks_measures = [ks_test(true_samples, data[row][i]) for i in
                xrange(num_L_trials)]
        plt.figure(1)
        plt.title('Leapfrog steps vs KS, eps=%f' % eps)
        plt.xlabel('Leapfrog steps')
        plt.ylabel('KS statistic')
        plt.grid(True)
        plt.plot(L_trials, ks_measures, 'bo--')
        #plt.show()
        plt.savefig('normal_distribution_accuracy_plots/eps=%.2f.png' % eps)
        plt.close()

def ks_test(s1, s2): #Kolmogorov-Smirnoff test 
    num_checkpoints = 99
    num_samples = len(s1)
    s1.sort()
    s2.sort()

    checkpoints = np.linspace(-2, 2, num_checkpoints)
    #print checkpoints
    cdf_1 = compute_cdf(s1, checkpoints)
    cdf_2 = compute_cdf(s2, checkpoints)
    #print cdf_1, cdf_2

    max_diff = 0
    for a,b in zip(cdf_1, cdf_2):
        max_diff = max(max_diff, abs(a-b))

    return max_diff / num_samples

def compute_cdf(s, checkpoints):
    num_checkpoints = len(checkpoints)
    num_samples = len(s)
    i = 0 #index in checkpoints
    j = 0 #index in samples
    cdf = []

    while True:
        while s[j] < checkpoints[i]:
            j += 1
            if j == num_samples:
                while len(cdf) < num_checkpoints:
                    cdf.append(num_samples)
                return cdf
        cdf.append(j)
        i += 1
        if i == num_checkpoints:
            while len(cdf) < num_checkpoints:
                cdf.append(j)
            return cdf
        #print cdf

if __name__ == "__main__":
    main()
