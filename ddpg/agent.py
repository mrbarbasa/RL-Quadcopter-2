import numpy as np

from ddpg.actor import Actor
from ddpg.critic import Critic
from ddpg.ou_noise import OUNoise
from ddpg.replay_buffer import ReplayBuffer

class DDPG():
    """Reinforcement Learning agent that learns using DDPG.

    DDPG is actually an actor-critic method, but the key idea is that 
    the underlying policy function used is deterministic in nature, 
    with some noise added in externally to produce the desired 
    stochasticity in actions taken.
    """
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.01  # for soft update of target parameters

        # Score tracker and learning parameters
        self.best_score = -np.inf
        self.score = -np.inf
        self.total_reward = 0.0
        self.count = 0

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0

        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        action = list(action + self.noise.sample()) # add some noise for exploration
        
        # Set the rotors to the same speed
        rotor_speed_mean = np.mean(action)
        action = [rotor_speed_mean for _ in action]
        return action # rotor speeds

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def learn(self, experiences):
        """Update policy and value parameters using the given batch of experience tuples.

        We will need two copies of each model - one local and one target. 
        This is an extension of the "Fixed Q Targets" technique from Deep Q-Learning, 
        and is used to decouple the parameters being updated (local model) from the ones
        that are producing target values (target model).

        Notice that after training over a batch of experiences, we could just copy our
        newly learned weights from the local model to the target model. However,
        individual batches can introduce a lot of variance into the process, so it's
        better to perform a soft update, controlled by the parameter tau.
        """
        # Keep track of the current score and best score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score

        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1]) # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        # We're currently only incorporating 1% of the local weights into the target weights
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
