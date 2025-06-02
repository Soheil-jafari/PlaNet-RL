import gym

def make_env(env_name='CartPole-v1', seed=0):
    """
    Creates and returns a Gym environment with a fixed seed for reproducibility.
    """
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
