__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2024  All rights reserved."

import numpy as np
from mcmc import MCMC


"""
    Implementation of Metropolis-Hastings Monte Carlo Markov Chain
    :param markov_model Markov model of transitional probabilities 
    :param likelihood_function Likelihood associated with a proposed distribution
    :param prior Class prior probability associated with the proposed distribution
    :param num_iterations Number of iterations for the random walk
    :param burn_in_ratio Percentage of number of iterations dedicated to burn-in steps
    :param sigma_delta Covariance or standard deviation used for each step theta -> theta_star
    @author Patrick Nicolas
"""


class MetropolisHastings(MCMC):
    from proposal_distribution import ProposalDistribution

    def __init__(self,
                 proposal: ProposalDistribution,
                 num_iterations: int,
                 burn_in_ratio: float,
                 sigma_delta: float = 0.2) -> None:
        assert 2 <= num_iterations <= 100000, f'Number of iterations {num_iterations} is out of bounds [2, 10000]'
        assert 0.0 < sigma_delta < 1.0, f'Sigma differential {sigma_delta} is out of bounds ]0.0, 1.0['
        assert 0.0 <= burn_in_ratio <= 0.5, f'Burn-in ratio {burn_in_ratio} is out of bounds [0.0, 0.5]'

        self.proposal = proposal
        self.num_iterations = num_iterations
        self.sigma_delta = sigma_delta
        self.burn_ins = int(num_iterations*burn_in_ratio)

    def sample(self, theta_0: float) -> (np.array, float):
        """
            :param theta_0 Initial value for the parameters
            :return Tuple of history of theta values after burn-in and ratio of number of accepted
                    new theta values over total number of iterations after burn-ins
        """
        num_valid_thetas = self.num_iterations - self.burn_ins
        theta_walk = np.zeros(num_valid_thetas)
        accepted_count = 0
        theta = theta_0    # 0.25
        theta_walk[0] = theta

        j = 0
        for i in range(self.num_iterations):
            theta_star = self.proposal.step(theta, self. sigma_delta)

            try:
                # Computes the prior for the current and next sample
                cur_prior = self.proposal.log_prior(theta)
                new_prior = self.proposal.log_prior(theta_star)

                # We only consider positive and non-null prior probabilities
                if cur_prior > 0.0 and new_prior > 0.0:
                    # We use the logarithm value for the comparison to avoid underflow
                    cur_log_posterior = self.proposal.log_posterior(theta, cur_prior)
                    new_log_posterior = self.proposal.log_posterior(theta_star, new_prior)

                    # Apply the selection criteria
                    if MetropolisHastings.__acceptance_rule(cur_log_posterior, new_log_posterior):
                        theta = theta_star
                        if i > self.burn_ins:
                            accepted_count += 1
                            theta_walk[j + 1] = theta_star
                            j += 1
                    else:
                        if i > self.burn_ins:
                            theta_walk[j + 1] = theta_walk[j]
                            j += 1

            except ArithmeticError as e:
                print(f'Arithmetic error: {e}')
            except ValueError as e:
                print(f'Value error: {e}')

        return theta_walk, float(accepted_count) / num_valid_thetas

        # --------------  Supporting methods -----------------------
    @staticmethod
    def __acceptance_rule(current: float, new: float) -> bool:
        residual = new - current
        return True if new > current else np.random.uniform(0, 1) < np.exp(residual)