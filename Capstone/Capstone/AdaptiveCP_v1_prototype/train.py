import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from env import AdaptiveCPEnv
from DQN import DQN

# 하이퍼파라미터
LR = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000

def train():
    env = AdaptiveCPEnv()
    device = torch.device("cpu")
    
    policy_net = DQN(env.state_dim, env.action_dim).to(device)
    target_net = DQN(env.state_dim, env.action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)
    criterion = nn.MSELoss()
    
    epsilon = EPSILON_START
    episodes = 1000
    
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(100):
            if random.random() < epsilon:
                action = random.randrange(env.action_dim)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
                
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)
                
                b_states = torch.FloatTensor(np.array(b_states)).to(device)
                b_actions = torch.LongTensor(b_actions).unsqueeze(1).to(device)
                b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1).to(device)
                b_next_states = torch.FloatTensor(np.array(b_next_states)).to(device)
                
                curr_q = policy_net(b_states).gather(1, b_actions)
                max_next_q = target_net(b_next_states).max(1)[0].unsqueeze(1)
                expected_q = b_rewards + (GAMMA * max_next_q)
                
                loss = criterion(curr_q, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)
        
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    # 모델 가중치 및 학습 보상 데이터 저장 (테스트에서 불러오기 위함)
    torch.save(policy_net.state_dict(), "dqn_adaptive_cp.pth")
    np.save("train_rewards.npy", np.array(episode_rewards))
    print("Training Complete. Model and Reward Data Saved.")

if __name__ == "__main__":
    train()