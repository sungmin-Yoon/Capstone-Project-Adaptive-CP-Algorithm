import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque

from env import AdaptiveCPEnv
from DQN import DQN

# ── 하이퍼파라미터 (중간보고서 Table 2 기준) ───────────────────────────────
LR             = 0.001
GAMMA          = 0.99
EPSILON_START  = 1.0
EPSILON_END    = 0.01
EPSILON_DECAY  = 0.995
BATCH_SIZE     = 64
MEMORY_SIZE    = 10_000
EPISODES       = 1_000
STEPS_PER_EP   = 100
TARGET_UPDATE  = 10          # Target Network 갱신 주기 (에피소드)


def train():
    env    = AdaptiveCPEnv()
    device = torch.device("cpu")
    print(f"[Train] Using device: {device}")

    policy_net = DQN(env.state_dim, env.action_dim).to(device)
    target_net = DQN(env.state_dim, env.action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    memory    = deque(maxlen=MEMORY_SIZE)

    epsilon = EPSILON_START
    episode_rewards = []

    for episode in range(EPISODES):
        state      = env.reset()
        total_reward = 0.0

        for step in range(STEPS_PER_EP):
            # ε-greedy 행동 선택
            if random.random() < epsilon:
                action = random.randrange(env.action_dim)
            else:
                with torch.no_grad():
                    q = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                action = q.argmax().item()

            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state         = next_state
            total_reward += reward

            # 미니배치 학습
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                b_s, b_a, b_r, b_ns, b_d = zip(*batch)

                b_s   = torch.FloatTensor(np.array(b_s)).to(device)
                b_a   = torch.LongTensor(b_a).unsqueeze(1).to(device)
                b_r   = torch.FloatTensor(b_r).unsqueeze(1).to(device)
                b_ns  = torch.FloatTensor(np.array(b_ns)).to(device)

                curr_q     = policy_net(b_s).gather(1, b_a)
                max_next_q = target_net(b_ns).max(1)[0].unsqueeze(1).detach()
                expected_q = b_r + GAMMA * max_next_q

                loss = criterion(curr_q, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)

        # Target Network 동기화
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"  Episode {episode:4d} | Reward: {total_reward:8.2f} | ε: {epsilon:.4f}")

    # 모델 및 학습 보상 저장
    torch.save(policy_net.state_dict(), "dqn_adaptive_cp.pth")
    np.save("train_rewards.npy", np.array(episode_rewards))
    print("\n[Train] Complete. Model saved → dqn_adaptive_cp.pth")


if __name__ == "__main__":
    train()
