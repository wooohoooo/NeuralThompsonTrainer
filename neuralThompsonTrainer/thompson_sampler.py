# AUTOGENERATED! DO NOT EDIT! File to edit: 01_thompson_sampler.ipynb (unless otherwise specified).

__all__ = ['GaussianBandit']

# Cell

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.cm as cm


class GaussianBandit(object):

    def __init__(self, num_options = 2, mean_prior = 0, std_prior = 1):
        """initialize BetaBandit"""
        self.num_options = num_options

        #setting the prior, either uninformative or user generated
#         if prior == None:
#             self.prior = np.ones(shape=(num_options,2))
#         else:
#             assert prior.shape == (num_options,2), f"the prior seems to have wrong dimensionality, please conform to (num_options, 2){(num_options,2)}"
#             self.prior = prior

        self.trials = np.zeros(shape=(num_options,))
        self.sum_x = np.zeros(shape=(num_options,))
        self.sum_x2 = np.zeros(shape=(num_options,))

        self.mean_prior = mean_prior
        self.std_prior = std_prior

    def choose_arm(self):
        """draw from arms.
        arm with the highest expected outcome wins.
        expected outcome is determined stochastically, so even an arm with bad
        outcome until now will have a chance of being drawn"""

        sampled_outcomes = []
        for i in range(self.num_options):

            if self.trials[i] > 1:
                mean = self.compute_online_mean(i)
                stdev = self.compute_online_std(i, mean)
            else:
                mean = 0
                stdev = 1


            dist = stats.norm(mean,stdev)

            sampled_outcome = dist.rvs()
            #print(sampled_outcome)

            sampled_outcomes += [sampled_outcome]

        return np.argmax(sampled_outcomes)

        #return(sampled_outcomes.argmax(sampled_outcomes))

    def update(self, arm, outcome):
        """update parameters of specific arm"""
        #count times arm has been drawn"""
        self.trials[arm] = self.trials[arm] +1
        #count number of successes on that arm"""

        # for decay factors: self.successes = self.successes *.99

        self.sum_x[arm] += outcome
        self.sum_x2[arm] += outcome*outcome


    def compute_online_mean(self, arm):
        return self.sum_x[arm] / (self.trials[arm])

    def compute_online_std(self, arm, mean = None):
        mean = mean or self.compute_online_mean(arm)
        #np max against degeneration)
        return np.max([np.sqrt((self.sum_x2[arm] / (self.trials[arm])) - (mean * mean)), 0.00001])




    def plot_params(self):
        """plot the distributions that underly the arms"""

        w = 10
        z = 5
        colors = iter(cm.rainbow(np.linspace(0, 1, self.num_options)))

        for k,i in enumerate(range(self.num_options)):
                color = next(colors)




                if self.trials[i] > 1:
                    mean = self.compute_online_mean(i)
                    stdev = self.compute_online_std(i, mean)
                else:
                    mean = 0
                    stdev = 1


                dist = stats.norm(mean,stdev)

                x = np.linspace(-6,6,100)
                y = dist.pdf(x)
                plt.plot(x,y,color=color,label="arm #%i"%(i+1))
                plt.fill_between(x,0,y,alpha=1/self.num_options,color=color)
                leg = plt.legend()
                plt.tight_layout


