import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
num_steps = 1500
obs = env.reset()

def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    num_states = (env.observation_space.high - env.observation_space.low)*\
        np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1 # I think this is where I will change for continuous?
    Q = np.random.uniform(low = -1, high = 1, size = (num_states[0], num_states[1], env.action_space.n))
    reward_list = []
    ave_reward_list = []

    reduction = (epsilon - min_eps)/episodes
    for i in range(episodes):
        done = False
        tot_reward, reward = 0,0
        state = env.reset()

        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
        while not done:
            if i >= (episodes - 20):
                env.render()
            
            # Determine next action to take (Using epsilon-greedy)
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            state2, reward, done, info = env.step(action)

            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)

            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
            else:
                delta = learning*(reward + discount*np.max(Q[state2_adj[0], state2_adj[1]]) - Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta

            tot_reward += reward
            state_adj = state2_adj

        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction
        
        # Track rewards
        reward_list.append(tot_reward)
        if (i+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
        if (i+1) % 100 == 0:
            print(f"Episode {i+1}: Average Reward: {ave_reward}")
    env.close()

    return ave_reward_list

rewards = QLearning(env, 0.2, 0.9, 0.8, 0, 1000000)

plt.plot(100*np.arange(len(rewards)) + 1, rewards)
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Average Reward vs. Episodes")
plt.savefig('rewards.jpg')
plt.close()
            






# for step in range(num_steps):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         env.reset()

# env.close()
# print(f"The initial observation is {obs}")
# random_action = env.action_space.sample()
# newobs, reward, done, info = env.step(random_action)
# print(f"The new observation is {newobs}")

# # print('\n\n')
# # print(f'Observation Space : {env.observation_space}')
# # print(f'Action Space : {env.action_space}')

# # print(env.observation_space.low)
# # print(env.observation_space.high)
# # print('\n\n')



# # for _ in range(1000):
# #     env.render()
# #     env.step(env.action_space.sample())
# env_screen = env.render(mode = "rgb_array")
# env.close()
# plt.imshow(env_screen)
