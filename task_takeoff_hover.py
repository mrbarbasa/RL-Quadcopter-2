import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent.

       This is the takeoff and hover task.
       Only the reward function has changed.
    """
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current z-axis pose and velocity of sim to return reward.

           Note that the highest reward is given when going 50v or more and 
           the agent has reached its target pose.
        """

        z_max_pose = 300
        current_pose = self.sim.pose[2]
        target_pose = self.target_pos[2]
        # The range is [0, 1], so let's multiply by 2 and subtract by 1
        # to normalize to [-1, 1]
        bias_pose = abs(current_pose - target_pose) / z_max_pose
        bias_pose = bias_pose * 2 - 1
        # We want the bias in position to be a negative reward
        bias_pose = -bias_pose

        v = self.sim.v[2]
        v_limit = 50
        v_norm = np.clip(v, -v_limit, v_limit) / v_limit # [-1, 1]

        # Check if the agent overshot its target; if so, start
        # penalizing the positive z-axis velocity
        if current_pose > target_pose:
            v_norm = -v_norm

        # Divide by the number of summands
        reward = (v_norm + bias_pose) / 2 # [-1, 1]

        return reward

    # Currently: action=rotor_speeds
    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            # By default, the episode is considered done if the time limit has 
            # been exceeded, or the quadcopter has travelled outside of the bounds 
            # of the simulation.
            done = self.sim.next_timestep(action) # update the sim pose and velocities

            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all) # Returns np.array of length 18
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        # Need to convert to a list or else the following is thrown:
        # `ValueError: zero-dimensional arrays cannot be concatenated`
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
