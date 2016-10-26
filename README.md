# HMC_sampler

Some experimentation with a Hamiltonian Monte Carlo sampler.

At the moment I am seeing how well one step of the leapfrog method can
approximate N(0,1). The answer seems to be not very accurately. To see this,
run basic_distributions.py to generate sample_matrix.pkl. Running plot_data.py
will then generate the plots found in normal_distribution_accuracy plots.
Unfortunately, these plots suggest there is no particularly clear relationship
between L (number of steps within a leapfrog step) and epsilon (step size) and
the accuracy of the distribution it generates.
