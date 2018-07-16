import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent.

       This is the takeoff and reach a target pose task.
       Only the reward and step functions have changed.
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

    def get_reward(self, target_reached):
        """Uses current z-axis velocity of sim to return reward."""

        v = self.sim.v[2]
        v_limit = 50
        v_norm = np.clip(v, -v_limit, v_limit) / v_limit # [-1, 1]
        reward = v_norm

        # Terminal state rewards
        if target_reached:
            # print('Target reached!', self.sim.pose[2])
            reward += 1.

        return reward

    # Currently: action=rotor_speeds
    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        target_reached = False
        pose_all = []
        for _ in range(self.action_repeat):
            # By default, the episode is considered done if the time limit has 
            # been exceeded, or the quadcopter has travelled outside of the bounds 
            # of the simulation.
            done = self.sim.next_timestep(action) # update the sim pose and velocities

            # End the episode if the agent has reached the target
            if not done:
                target_reached = self.sim.pose[2] >= self.target_pos[2]
                done = target_reached

            reward += self.get_reward(target_reached)
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
