import numpy as np
import matplotlib.pyplot as plt

class PomdpEnv():
    """
    Base class for POMDP environments.
    """
    def __init__(self):
        raise NotImplementedError("This is a base class. Please implement the methods in a subclass.")

    def reset(self):
        raise NotImplementedError("Reset method must be implemented in the subclass.")

    def step(self, action):
        raise NotImplementedError("Step method must be implemented in the subclass.")

    def render(self):
        raise NotImplementedError("Render method must be implemented in the subclass.")

class CliffWalk(PomdpEnv):
    """
    Partially Observable Cliff Walk Environment
    """
    def __init__(self, n=3, m=5, self_transition_prob=0.2, gamma=0.9):
        self.n = n # Number of rows
        self.m = m # Number of columns
        self.state_dim = n * m # Total number of states
        self.self_transition_prob = self_transition_prob # Probability of staying in the same state
        self.action_space = [0, 1, 2, 3] # left, up, right, down
        self.gamma = gamma

        self.generic_reward = -1.0
        self.cliff_reward   = -10.0
        self.target_reward  = 0.0

        self.state = None
        self.belief_state = None
        self.done = False
        
        self.tp_matrix   = self.init_tp_matrix()
        self.reward_vec  = self.init_reward_vec()
        self.obs_matrix  = self.init_observation_matrix()

        self.obs_dim = self.obs_matrix.shape[1]  # Number of unique observations

        self.reset()

    def init_tp_matrix(self):
        """
        Transition Probability Matrix (TPM) for the environment.
        the TPM is a 3D tensor of size [|actions|, |states|, |states|]
        """
        tpm = np.zeros((len(self.action_space), self.state_dim, self.state_dim))
        for x in range(self.m):
            for y in range(self.n):
                current_state_index = x + y * self.m
                for action in self.action_space:
                    # Terminal states have 0 outgoing transitions
                    if y == 0 and x > 0:
                        continue
                    
                    # Non-terminal states
                    else:
                        if action == 2: # right
                            x_new = min(x+1, self.m-1)
                            y_new = y
                        elif action == 1: # up
                            y_new = min(y+1, self.n-1)
                            x_new = x
                        elif action == 3: # down
                            y_new = max(y-1, 0)
                            x_new = x
                        elif action == 0: # left
                            x_new = max(x-1, 0)
                            y_new = y

                        target_state_index = x_new + y_new * self.m
                        # add probability of moving to the target state
                        tpm[action, current_state_index, target_state_index] += (1 - self.self_transition_prob)
                        # add probability of staying in the same state
                        tpm[action, current_state_index, current_state_index] += self.self_transition_prob

        return tpm

    def init_reward_vec(self):
        """
        Reward vector for the environment. size is [|states|]
        The reward is -1 for all states except the goal state (0, m-1) which has a reward of 0.
        and the cliff states which have a reward of -100.
        """
        reward_vec = np.full((self.state_dim), self.generic_reward)
        for x in range(self.m):
            for y in range(self.n):
                if (x == self.m - 1 and y == 0):
                    reward_vec[x + y * self.m] = self.target_reward
                elif y == 0 and x > 0 and x < self.m - 1:
                    reward_vec[x + y * self.m] = self.cliff_reward
        
        return reward_vec

    def init_observation_matrix(self):
        """
        Observation matrix for the environment. size is [|states|, |observations|]
        start and end states are revealed as seperate observations, 
        otherwise only vertical position is revealed.
        """
        obs_matrix = np.zeros((self.state_dim, self.n+2))
        for x in range(self.m):
            for y in range(self.n):
                current_state_index = x + y * self.m
                # Start state has unique observation
                if x == 0 and y == 0:
                    obs_matrix[current_state_index, 0] = 1.0
                # Goal state has unique observation
                elif y == 0 and x == self.m - 1:
                    obs_matrix[current_state_index, 1] = 1.0
                # Otherwise, only vertical position is revealed
                else:
                    obs_matrix[current_state_index, y + 2] = 1.0
        
        ## Fully observable case (for debugging purposes)
        # obs_matrix = np.zeros((self.state_dim, self.state_dim))
        # for x in range(self.m):
        #     for y in range(self.n):
        #         current_state_index = x + y * self.m
        #         obs_matrix[current_state_index, current_state_index] = 1.0

        return obs_matrix
    
    def get_optimal_policy(self, epsilon=0.0):
        """
        Get the optimal policy for the environment, optionally with epsilon-greedy exploration.
        """
        pi = np.zeros((self.state_dim, len(self.action_space)))
        for x in range(self.m):
            for y in range(self.n):
                current_state_index = x + y * self.m
                # Start state, move up
                if x == 0 and y == 0:
                    pi[current_state_index, 1] = 1.0
                # Any non-cliff state, not at the right edge, move right
                elif y > 0 and x < self.m - 1:
                    pi[current_state_index, 2] = 1.0
                # in terminal states (cliff and goal) all actions are equally likely
                elif y == 0 and x > 0:
                    pi[current_state_index, :] = 0.25
                # right edge, move down
                else:
                    pi[current_state_index, 3] = 1.0

        # Add epsilon-greedy exploration
        if epsilon > 0.0:
            for state_index in range(self.state_dim):
                pi[state_index] = (1 - epsilon) * pi[state_index] + (epsilon / 4)
        
        return pi

    def get_value_function(self, policy):
        """
        Calculate the value function under a given policy on the latent state space.
        """
        # Get the marginal transition probability matrix under the policy
        P_pi = np.einsum('sa, asn -> sn', policy, self.tp_matrix)
      
        # Get the reward vector under the policy using V = (I - P_pi)^-1 @ R
        I = np.eye(self.state_dim)
        V_pi = np.linalg.solve(I - self.gamma*P_pi, self.reward_vec)

        return np.round(V_pi, 2)
    
    def get_q_value_function(self, policy):
        """
        Calculate the Q-value function under a given policy on the latent state space.
        """
        # Get value function under the policy
        V_pi = self.get_value_function(policy)

        # Calculate Q-values via Q(s, a) = R(s) + sum_{s'} P(s'|s, a) V(s')
        Q_pi = np.zeros((self.state_dim, 4))
        Q_pi += self.reward_vec[:, None]
        Q_pi += self.gamma*np.einsum('asn, n -> sa', self.tp_matrix, V_pi)

        return np.round(Q_pi, 2)

    def reset(self):
        self.done = False

        # Reset state to the start position (0, 0)
        self.state = np.zeros(self.state_dim)
        self.state[0] = 1
        
        # Observation is the unique start position
        observation = np.zeros(self.obs_dim)
        observation[0] = 1

        # Reward is -0.1 at the start position
        reward = self.generic_reward

        # Initialize the fully resolved belief state at the start position
        self.belief_state = np.zeros(self.state_dim)
        self.belief_state[0] = 1
        
        return self.state, observation, reward, self.belief_state, self.done
    

    def step(self, action):
        if self.done:
            raise ValueError("Episode is done. Please reset the environment.")
        
        # Update state via s_t ~ p(s_t | s_{t-1}, a_{t-1})
        state_probs = self.state @ self.tp_matrix[action]
        state_index = np.random.choice(self.state_dim, p=state_probs)
        state       = np.zeros(self.state_dim)
        state[state_index] = 1
        
        # Reward is a scalar value s_t @ r_t
        reward      = self.reward_vec @ state
        
        # Observation index is given by o_t ~ p(o_t | s_t)
        obs_probs   = self.obs_matrix[state_index]
        obs_index   = np.random.choice(self.obs_dim, p=obs_probs)
        observation = np.zeros(self.obs_dim)
        observation[obs_index] = 1

        # Belief update is via p(x_t|b_t) \propto p(o_t | s_t) \sum_{s_{t-1}} p(s_t | s_{t-1}, a_{t-1}) p(x_{t-1}|b_{t-1})
        belief_state = self.belief_state @ self.tp_matrix[action]
        belief_state *= self.obs_matrix[:, obs_index]
        belief_state /= np.sum(belief_state)
        self.belief_state = belief_state

        # Check if the episode is done
        if reward == self.target_reward or reward == self.cliff_reward:
            self.done = True
        self.state = state
        
        return state, observation, reward, belief_state, self.done
    
    def render(self):
        """
        Render the environement"
        """
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # render the position of the agent
        axs[0].imshow(self.state.reshape((self.n, self.m)), cmap='gray', vmin=0, vmax=1)
        axs[0].set_title("Agent Position")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].invert_yaxis()
        # render the belief state
        axs[1].imshow(self.belief_state.reshape((self.n, self.m)), cmap='gray', vmin=0, vmax=1)
        axs[1].set_title("Belief State")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].invert_yaxis()
        plt.show()


class Episode():
    """
    Class for handling data from a POMDP episode
    """
    def __init__(self):
        self.states = []
        self.observations = []
        self.rewards = []
        self.actions = []
        self.belief_states = []

        self.attached = False
        
    def add_step(self, state, observation, reward, action, belief_state):
        """
        Add a step to the episode data

        :param state: Current state of the environment, one-hot encoded numpy vector
        :param observation: Current observation of the environment, one-hot encoded numpy vector
        :param reward: Reward received from the environment, scalar
        :param action: Action taken in the environment, one-hot encoded numpy vector
        :param belief_state: Current belief state of the environment, vector of probabilities
        """
        self.states.append(state)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.actions.append(action)
        self.belief_states.append(belief_state)

    def finish_episode(self):
        """
        Finish the episode and attach the data
        """
        self.attach()

    def attach(self):
        """
        Turn the lists into numpy arrays for easier handling
        """
        if not self.attached:
            self.states         = np.array(self.states)
            self.observations   = np.array(self.observations)
            self.rewards        = np.array(self.rewards)    
            self.actions        = np.array(self.actions)
        
            # Create a history from column-concatenating the observations and actions
            self.history        = np.concatenate((self.observations, self.actions), axis=1)
            self.belief_states  = np.array(self.belief_states)
           
            self.attached = True

    def render(self):
        """
        Render the episode data
        """
        self.attach()
        
        fig, axs = plt.subplots(5, 1, figsize=(8, 10))
        
        # Render the states
        axs[0].imshow(self.states.T, aspect='auto', cmap='gray')
        axs[0].set_title("States")
        axs[0].set_xlabel("Time Step")
        axs[0].set_ylabel("State Index")

        # Render the observations
        axs[1].imshow(self.observations.T, aspect='auto', cmap='gray')
        axs[1].set_title("Observations")
        axs[1].set_xlabel("Time Step")
        axs[1].set_ylabel("Observation Index")

        # Render the actions
        axs[2].imshow(self.actions.T, aspect='auto', cmap='gray')
        axs[2].set_title("Actions")
        axs[2].set_xlabel("Time Step")
        axs[2].set_ylabel("Action Index")

        # Render the rewards
        axs[3].plot(self.rewards, marker='o', linestyle='-', color='b')
        axs[3].set_title("Rewards")
        axs[3].set_xlabel("Time Step")

        # Render the belief states
        axs[4].imshow(self.belief_states.T, aspect='auto', cmap='gray')
        axs[4].set_title("Belief States")
        axs[4].set_xlabel("Time Step")
        axs[4].set_ylabel("State Index")
        
        plt.tight_layout()
        plt.show()


def collect_episodes(env:PomdpEnv, policy:np.array, num_episodes:int) -> list[Episode]:
    """
    Collect episodes from the environment using a given policy.
    Args:
        env:            The environment to collect episodes from.
        policy:         A callable that takes the current state and returns an action.
        num_episodes:   Number of episodes to collect.
    Returns:
        A list of "Episode" objects.
    """
    episodes = []
    for _ in range(num_episodes):
        # Initialize a new episode
        episode = Episode()

        # Reset the environment, and emit the first data points
        state, observation, reward, belief, done = env.reset()
        prev_action = np.zeros(len(env.action_space))  # No action taken yet
        
        while not done:
            # Add step to the episode
            episode.add_step(state, observation, reward, prev_action, belief)
            
            # Sample state for the policy from the belief state, and act as if in that state
            act_as_if_state = np.random.choice(env.n * env.m, p=belief)
            action = np.random.choice(env.action_space, p=policy[act_as_if_state])
            prev_action = np.zeros(len(env.action_space))
            prev_action[action] = 1
            
            # Step the environment with the chosen action
            state, observation, reward, belief, done = env.step(action)

        # update the episode with the final states, finish, and append it to the list
        episode.add_step(state, observation, reward, prev_action, belief)
        episode.finish_episode()
        episodes.append(episode)
    
    return episodes  
        
def monte_carlo_state_values(episodes: list[Episode], gamma=0.9):
    """
    Monte Carlo estimation of V(s) for one-hot discrete states.

    Args:
        episodes:       list of "Episode" objects
        gamma:          discount factor

    Returns:
        state_values:   array of shape (num_states,) with estimated V(s)
    """
    assert len(episodes) > 0, "No episodes provided for Monte Carlo estimation."
    assert isinstance(episodes, list), "Episodes must be a list of Episode objects."
    assert all(isinstance(ep, Episode) for ep in episodes), "All items in the list must be Episode objects."
    assert all(ep.attached for ep in episodes), "All episodes must be attached (data converted to numpy arrays)."

    num_states = episodes[0].states.shape[1]
    state_returns = np.zeros(num_states)
    state_counts = np.zeros(num_states)

    for ep in episodes:
        rewards = ep.rewards
        states = ep.states

        G = 0.0
        returns = [0.0] * len(rewards)
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns[t] = G

        for t, state in enumerate(states):
            state_idx = np.argmax(state).item()  # Get index from one-hot
            state_returns[state_idx] += returns[t]
            state_counts[state_idx] += 1

    # Avoid divide-by-zero
    nonzero_mask = state_counts > 0
    state_values = np.zeros(num_states)
    state_values[nonzero_mask] = state_returns[nonzero_mask] / state_counts[nonzero_mask]

    return np.round(state_values, 2)