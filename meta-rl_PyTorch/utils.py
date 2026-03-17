import tensorflow as tf
import numpy as np
import pyDOE
from scipy.stats.distributions import truncnorm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import scipy

"""
Utilities
"""

EPS = 1e-8

def placeholder(dim=None):
    return tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

import tensorflow as tf

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.compat.v1.keras.layers.Dense(
            units=h,
            activation=activation
        )(x)
    return tf.compat.v1.keras.layers.Dense(
        units=hidden_sizes[-1],
        activation=output_activation
    )(x)

def get_vars(scope):
    return [x for x in tf.compat.v1.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(256,256), activation=tf.nn.relu, 
                     output_activation=tf.tanh, action_space=None):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.compat.v1.variable_scope('pi'):
        pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    with tf.compat.v1.variable_scope('q'):
        q = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.compat.v1.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, q, q_pi

"""
Actor-Meta-policy (for continuous action only)
"""
def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):

    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]
    
    #mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    mu = act_limit *mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.compat.v1.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu #+ tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


def mlp_actor_critic_ppo(x, a, hidden_sizes=(128, 128), activation=tf.nn.tanh, output_activation=None, action_space=None):
    with tf.compat.v1.variable_scope('pi'):
        pi, logp, logp_pi = mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space)
    with tf.compat.v1.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, v


"""
Memory replay buffer
"""

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


"""
Sample environment parameters

"""
def sample_param(num_sample):
    # Thermal capacitance, W/C
    mu_c_env = 3.1996e6
    mu_c_air = 3.5187e5
    # Thermal resistance, J/W
    mu_r_rc = 0.00706
    mu_r_oe = 0.02707
    mu_r_er = 0.00369

    #sample the properties
    design = pyDOE.lhs(5, samples = num_sample)
    mu = [mu_c_env, mu_c_air, mu_r_rc, mu_r_oe, mu_r_er]
    sigma = [mu_c_env/1.5, mu_c_air/1.5, mu_r_rc/1.5, mu_r_oe/1.5, mu_r_er/1.5]
    lower = np.array([0.5e6, 0.5e5, 0.1e-3, 0.2e-2, 0.1e-3])
    upper = np.array([10e6, 10e5, 15e-3, 5e-1, 10e-3])
    a = (lower - mu)/sigma
    b = (upper - mu)/sigma
    for i in range(5):
        design[:, i] = truncnorm(a[i], b[i], loc=mu[i], scale=sigma[i]).ppf(design[:, i])
 
    return design


"""
Generate relevant plots

"""

def custom_plot(T_air, time, T_out, Q_SG, action_list, energy_list, 
                penalty_list, lb_list, ub_list, 
                idx, folder):

    sns.set_style("whitegrid")
    font = 15
    matplotlib.rc('xtick', labelsize = font) 
    matplotlib.rc('ytick', labelsize = font)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), dpi = 300)
    axs[0, 0].plot(time, T_air)
    axs[0, 0].plot(time, lb_list, color = 'orange', label = 'lower and upper bound')
    axs[0, 0].plot(time, ub_list, color = 'orange')
    axs[0, 1].plot(time, action_list)
    #axs[0, 1].plot(time, actionh_list)
    axs[1, 0].plot(time, T_out)
    axs[1, 1].plot(time, Q_SG)
    
    axs[0, 0].set_ylim([12, 30])
    axs[0, 1].set_ylim([0, 1])
    axs[1, 0].set_ylim([-30, 40])
    axs[1, 1].set_ylim([0, 1100])  
    
    axs[0, 0].set_ylabel("Indoor Air Temperature ($^\circ$C)", fontsize = font)
    axs[0, 1].set_ylabel("Thermal Input (W)", fontsize = font)
    axs[1, 0].set_ylabel("Outdoor Air Temperature ($^\circ$C)", fontsize = font)
    axs[1, 1].set_ylabel("Solar Heat Gain (W)", fontsize = font)
    
    axs[0, 0].set_xlabel("Time (hr)", fontsize = font)
    axs[0, 1].set_xlabel("Time (hr)", fontsize = font)
    axs[1, 0].set_xlabel("Time (hr)", fontsize = font)
    axs[1, 1].set_xlabel("Time (hr)", fontsize = font)
    
    axs[0, 0].legend(loc = 'best')
    
    fig.suptitle("Test Sample Env #%i" %idx, fontsize = font)    
    fig.savefig("./plots/"+str(folder)+"/ddpg_sample_env_policy_{}.png".format(idx))
    plt.close(fig)



"""
A buffer for storing trajectories experienced by a PPO agent interacting
with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
for calculating the advantages of state-action pairs.
"""

class PPOBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        #assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf), 
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]
