import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process.

    It essentially generates random samples from a Gaussian (Normal) distribution, 
    but each sample affects the next one such that two consecutive samples are 
    more likely to be closer together than further apart. In this sense, the 
    process in Markovian in nature.

    This process adds some noise to our actions, in order to encourage 
    exploratory behavior. But since our actions translate to force and torque 
    being applied to a quadcopter, we want consecutive actions to not vary wildly.
    This process helps reduce variation in consecutive actions.

    Besides the temporally correlated nature of samples, the other nice thing 
    about the OU process is that it tends to settle down close to the specified 
    mean over time. When used to generate noise, we can specify a mean of zero, 
    and that will have the effect of reducing exploration as we make progress 
    on learning the task.
    """

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
