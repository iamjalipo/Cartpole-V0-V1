import gym
e = gym.make('CartPole-v0')
print(e.action_space)
print(e.observation_space)
obs = e.reset()
print(obs)