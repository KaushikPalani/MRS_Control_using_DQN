from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from two_carrier_env import TwoCarrierEnv

# The env is imported using a class in this implementation
# Action, Observation and step values can be imported directly, if implemented through installed gym env
# Example 
# # import gym
# # env = gym.make('gym_ae5117:TriPuller-v0')
# # https://github.com/irasatuc/gym-ae5117

Action_space = 4            
Observation_space = 3       

class DqnAgent():
    def __init__(self):
        self.model = self.Construct_Neural_Network()
        self.target_model = self.Construct_Neural_Network()
        self.model_location = '..../Model'
        # Checkpoints to store the models
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), net=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, 'checkpoints', max_to_keep=20)
        # self.load_checkpoint()
        
    def policy(self, state, epsilon):
        # Epsilon greed 
        if np.random.random() < epsilon: return np.random.randint(0, Action_space)
        input_states = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        Qval = self.model(input_states)
        action = np.argmax(Qval.numpy()[0], axis=0)
        return action
    
    def Construct_Neural_Network(self):
        model = Sequential()
        model.add(Dense(300, input_dim=Observation_space, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(300, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(Action_space, activation='linear', kernel_initializer='he_uniform'))
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0002), loss='mse')
        return model
    
    def train(self, batch):
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = batch
        current_state_Qval_batch = self.model(state_batch)
        target_Qval_batch = np.copy(current_state_Qval_batch)
        next_state_Qval_batch = self.target_model(next_state_batch)
        max_next_state_Qval_batch = np.amax(next_state_Qval_batch, axis=1)
        for i in range(state_batch.shape[0]):
            target_Qval_batch[i][action_batch[i]] = reward_batch[i] if done_batch[i] else (reward_batch[i] + 0.99 * max_next_state_Qval_batch[i])
        # Train the Neural Network
        result = self.model.fit(x=state_batch, y=target_Qval_batch, workers=-1, use_multiprocessing=True)
        # self.save_checkpoint()
        return result.history['loss']
    
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_checkpoint(self):
        self.checkpoint_manager.save()

    def load_checkpoint(self):
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        
    def save_model(self):
        tf.saved_model.save(self.q_net, self.model_location)
  
    def load_model(self):
        self.q_net = tf.saved_model.load(self.model_location)


class ReplayBuffer:
    def __init__(self):
        self.Replay_memory = deque(maxlen=1000000)

    def Append_experience(self, state_c0, state_c1, next_state_c0, next_state_c1, reward, action, done):
        self.Replay_memory.append((state_c0, next_state_c0, reward, action, done))
        self.Replay_memory.append((state_c1, next_state_c1, reward, action, done))

    def Sample_minibatch(self):
        batch_size = min(Batch_size, len(self.Replay_memory))
        minibatch = random.sample(self.Replay_memory, batch_size)
        current_state_batch, next_state_batch, action_batch, reward_batch, done_batch = [], [], [], [], []
        for experience in minibatch:
            current_state_batch.append(experience[0])
            next_state_batch.append(experience[1])
            reward_batch.append(experience[2])
            action_batch.append(experience[3])
            done_batch.append(experience[4])
        return np.array(current_state_batch), np.array(next_state_batch), action_batch, reward_batch, done_batch


def Collect_experiences():
    state_c0, state_c1 = env.reset()
    done = False
    total_reward=0.0
    cnt=0
    while not done:
        action1 = agent.policy(state_c0, epsilon)
        action2 = agent.policy(state_c1, epsilon)
        next_state_c0, next_state_c1, reward, done, info = env.step([action1,action2])
        buffer.Append_experience(state_c0, state_c1, next_state_c0, next_state_c1, reward, [action1, action2], done)
        total_reward += reward
        state_c0 = next_state_c0
        state_c1 = next_state_c1
        cnt+=1
        if cnt>=Steps_per_episode:
            break
    return total_reward

def AgentTraining(episode_count, epsilon):
    total_reward_per_episode = Collect_experiences()
    Minibatch = buffer.Sample_minibatch()          
    # Train the DQNetwork
    loss = agent.train(Minibatch)
    # Store the rewards for plot 
    rewards.append(total_reward_per_episode)
    Average_rewards.append(sum(rewards[-Average_rewards_over:])/Average_rewards_over)
    # Evaluate the DQN Agent 
    if episode_count > Start_evaluation_episode: 
        average_eval_reward = Evaluate_DQNAgent()
        eval_rewards.append(average_eval_reward)
        # Save the model if the average reward is more than escape reward in the environment
        if eval_rewards[-1]>200: agent.save_checkpoint() # Escape reward in env = 200 (max value)
    # Plot interval
    if episode_count % Plot_every == 0: Plot(rewards, Average_rewards)  
    # Epsilon decay
    if episode_count > Warm_up_episodes: epsilon = max(epsilon * epsilon_decay, epsilon_min) 
    if episode_count % Update_target_every == 0: agent.update_target_network()
    return epsilon
        
def Evaluate_DQNAgent():
    individual_rewards = []
    Episodes_to_avg = 5
    for i in range(Episodes_to_avg): 
        c0_position, c1_position = env.reset()
        done = False
        episode_reward = 0.0
        for j in range(Steps_per_episode):
            current_state_c0 = np.float32(c0_position.copy())
            current_state_c1 = np.float32(c1_position.copy())
            # Query action
            action1 = agent.policy(current_state_c0, 0)
            action2 = agent.policy(current_state_c1, 0)
            c0_position, c1_position, reward, done, _ = env.step([action1, action2])
            episode_reward += reward
            if done:
                break
        individual_rewards.append(episode_reward)
    average_reward = sum(individual_rewards)/Episodes_to_avg
    return average_reward
       
def Plot(rewards, Average_rewards):   
    plt.plot(rewards, '0.8')
    plt.plot(Average_rewards, 'r', label='Average reward')    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('DQN')
    # # Calculate the trend
    # x = range(len(values))
    # try:
    #     z = np.polyfit(x, values, 1)
    #     p = np.poly1d(z)
    #     plt.plot(x,p(x),"--", label='trend')
    # except:
    #     print('')
    plt.show()    

env = TwoCarrierEnv()
agent = DqnAgent()
buffer = ReplayBuffer()

# Random seed
RANDOM_SEED = 0
tf.random.set_seed(RANDOM_SEED)
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)

# Hyper parameters
epsilon=1
epsilon_decay=0.999
epsilon_min=0.1

Start_evaluation_episode = 10_000
Plot_every = 300
Warm_up_episodes = 50
Episodes_to_train = 15_000
Average_rewards_over = 200
Steps_per_episode = 1000
Update_target_every = 2
Batch_size = 1000

rewards = []
Average_rewards = []
eval_rewards=[]

# Train DQN Agent 
for episode_count in range(Episodes_to_train): 
    epsilon = AgentTraining(episode_count, epsilon)
    print(episode_count, epsilon)
# Save model
agent.save_model()


# Visualize and verify trained model 
c0_position, c1_position = env.reset()
epsilon=0
for step in range(Steps_per_episode):
    env.render(step)
    current_state_c0 = np.float32(c0_position.copy())
    current_state_c1 = np.float32(c1_position.copy())
    # Query action
    action1 = agent.policy(current_state_c0, epsilon)
    action2 = agent.policy(current_state_c1, epsilon)
    # Perform the step based on the action 
    c0_position, c1_position, reward, done, info = env.step([action1, action2])    
    print(reward)
    if done:
        break



