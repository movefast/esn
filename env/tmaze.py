from copy import copy, deepcopy

import numpy as np

from env.environment import BaseEnvironment


class MazeEnvironment(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        self.rows, self.cols = self.maze_dim = env_info.get("maze_dim", [7, 7])
        self.obstacles_locs = env_info.get("obstacles", [])
        self.start_state = env_info.get("start_state", [0, 0])
        self.end_states = env_info.get("end_states", [[0, 0]])
        self.is_start_signal_learned = env_info.get("start_signal", False)
        self.current_state = None
        self.reward_obs_term = [0.0, None, False]
        self.rand_generator = np.random.RandomState(env_info.get("seed", 47))

    def env_start(self, keep_history=False):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.keep_history = keep_history
        self.current_state = self.start_state
        self.is_even = self.rand_generator.randint(10) % 2 == 0
        self.reward_obs_term = self.get_state_features(self.current_state)
        self.steps = 0
        self.obstacles = deepcopy(self.obstacles_locs)

        if self.is_even:
            self.rewards = {
                (2,1):1.,
                (0,1):-1.,
                (2,self.cols-1):10.,
                (0,self.cols-1):-10.,
            }
        else:
            self.rewards = {
                (2,1):-1.,
                (0,1):1.,
                (2,self.cols-1):-10.,
                (0,self.cols-1):10.,
            }

        if self.keep_history:
            # initialize a gridview of the environment
            self.update_grid()
            self.history = [copy(self.grid)]
        return self.reward_obs_term

    def out_of_bounds(self, row, col):
        """check if current state is within the gridworld and return bool"""
        if row < 0 or row > self.maze_dim[0]-1 or col < 0 or col > self.maze_dim[1]-1:
            return True
        else:
            return False

    def is_obstacle(self, row, col):
        """check if there is an obstacle at (row, col)"""
        if [row, col] in self.obstacles:
            return True
        else:
            return False

    def get_observation(self, state):
        return state[0] * self.maze_dim[1] + state[1]

    def get_state_features(self, state):
        if state == self.start_state or state == [1,1]:
            if self.is_even:
                if self.is_start_signal_learned:
                    bit = 1.1
                else:
                    bit = 2
            else:
                bit = 1
        else:
            bit = 0
        return [self.get_observation(state), bit]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        reward = 0.0
        is_terminal = False
        row, col = self.current_state

        if self.keep_history:
            self.grid[row, col] = 0

        # update current_state with the action (also check validity of action)
        if action == 0:  # left
            if not (self.out_of_bounds(row, col-1) or self.is_obstacle(row, col-1)):
                self.current_state = [row, col-1]

        elif action == 1:  # right
            if not (self.out_of_bounds(row, col+1) or self.is_obstacle(row, col+1)):
                self.current_state = [row, col+1]

        elif action == 2:  # up
            if not (self.out_of_bounds(row+1, col) or self.is_obstacle(row+1, col)):
                self.current_state = [row+1, col]

        elif action == 3:  # down
            if not (self.out_of_bounds(row-1, col) or self.is_obstacle(row-1, col)):
                self.current_state = [row-1, col]

        # terminate if goal is reached
        if self.current_state in self.end_states:
            is_terminal = True
        else:
            self.steps += 1

        if tuple(self.current_state) in self.rewards:
            reward = self.rewards[tuple(self.current_state)]
            self.rewards.pop(tuple(self.current_state))
        else:
            reward = 0

        self.reward_obs_term = [reward, self.get_state_features(self.current_state), is_terminal]
        if self.keep_history:
            row, col = self.current_state
            self.grid[row, col] = 100
            self.history.append(copy(self.grid))
        return self.reward_obs_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        current_state = None
        self.history = []
        self.grid = np.zeros(self.maze_dim)

    def update_grid(self):
        self.grid = np.zeros(self.maze_dim)
        self.grid[self.current_state[0], self.current_state[1]] = 100
        for locs, val in zip([self.end_states, self.obstacles], [-1, -100]):
            for row, col in locs:
                self.grid[row, col] = val
        self.grid = np.ma.masked_where(self.grid == -100, self.grid)

