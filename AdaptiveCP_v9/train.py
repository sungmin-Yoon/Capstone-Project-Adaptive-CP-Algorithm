import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque

from env import AdaptiveCPEnv
from DQN import DQN

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────
LR             = 0.001
GAMMA          = 0.99
EPSILON_START  = 1.0
EPSILON_END    = 0.01
EPSILON_DECAY  = 0.995
BATCH_SIZE     = 64
# [v9] 리플레이 버퍼 확대: 10,000 → 20,000
# 더 다양한 채널 환경(LOS/NLOS 조합)의 경험을 보존하여 정책 안정성 향상
MEMORY_SIZE    = 20_000
EPISODES       = 2_000
STEPS_PER_EP   = 100
TARGET_UPDATE  = 10

# NLOS 학습 설정
NLOS_EPISODE_RATIO = 0.4
NLOS_STEPS         = 10


def train():
    env    = AdaptiveCPEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    print(f"[Train] state_dim={env.state_dim}, action_dim={env.action_dim}")
    print(f"[Train] Episodes={EPISODES}, Steps/ep={STEPS_PER_EP}")
    print(f"[Train] Memory={MEMORY_SIZE}  [v9: 10,000 → 20,000]")

    policy_net = DQN(env.state_dim, env.action_dim).to(device)
    target_net = DQN(env.state_dim, env.action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPISODES, eta_min=1e-4
    )

    criterion = nn.MSELoss()
    memory    = deque(maxlen=MEMORY_SIZE)

    epsilon         = EPSILON_START
    episode_rewards = []

    for episode in range(EPISODES):
        state        = env.reset()
        total_reward = 0.0

        use_nlos   = random.random() < NLOS_EPISODE_RATIO
        nlos_start = random.randint(10, STEPS_PER_EP - NLOS_STEPS - 5) if use_nlos else -1
        nlos_end   = nlos_start + NLOS_STEPS if use_nlos else -1

        for step in range(STEPS_PER_EP):
            is_nlos = use_nlos and (nlos_start <= step < nlos_end)

            if random.random() < epsilon:
                action = random.randrange(env.action_dim)
            else:
                with torch.no_grad():
                    q = policy_net(
                        torch.FloatTensor(state).unsqueeze(0).to(device))
                action = q.argmax().item()

            next_state, reward, done, _ = env.step(action, nlos_event=is_nlos)
            memory.append((state, action, reward, next_state, done))
            state         = next_state
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                b_s, b_a, b_r, b_ns, b_d = zip(*batch)

                b_s  = torch.FloatTensor(np.array(b_s)).to(device)
                b_a  = torch.LongTensor(b_a).unsqueeze(1).to(device)
                b_r  = torch.FloatTensor(b_r).unsqueeze(1).to(device)
                b_ns = torch.FloatTensor(np.array(b_ns)).to(device)

                curr_q     = policy_net(b_s).gather(1, b_a)
                max_next_q = target_net(b_ns).max(1)[0].unsqueeze(1).detach()
                expected_q = b_r + GAMMA * max_next_q

                loss = criterion(curr_q, expected_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)
        scheduler.step()

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            nlos_tag = " [NLOS]" if use_nlos else ""
            print(f"  Ep {episode:4d}{nlos_tag} | "
                  f"Reward: {total_reward:8.2f} | ε: {epsilon:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(policy_net.state_dict(), "dqn_adaptive_cp.pth")
    np.save("train_rewards.npy", np.array(episode_rewards))
    print("\n[Train] Complete. Model saved → dqn_adaptive_cp.pth")


if __name__ == "__main__":
    train()
