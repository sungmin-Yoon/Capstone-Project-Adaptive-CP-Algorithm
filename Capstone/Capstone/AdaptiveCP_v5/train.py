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
MEMORY_SIZE    = 10_000
EPISODES       = 1_000
STEPS_PER_EP   = 100
TARGET_UPDATE  = 10       # Target Network 갱신 주기 (에피소드)

# ── v5: NLOS 학습 설정 ────────────────────────────────────────────────────
# 에피소드 내에서 NLOS 급변 구간을 명시적으로 포함
# NLOS_PROB: 한 에피소드 내 특정 스텝이 NLOS 급변으로 시작할 확률
NLOS_EPISODE_RATIO = 0.4   # 전체 에피소드 중 NLOS 구간을 포함하는 비율
NLOS_STEPS         = 10    # NLOS 구간 지속 스텝 수


def train():
    env    = AdaptiveCPEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    policy_net = DQN(env.state_dim, env.action_dim).to(device)
    target_net = DQN(env.state_dim, env.action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    # [v5] LR 스케줄러: 500 에피소드 이후 LR을 0.1배로 감소
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    criterion = nn.MSELoss()
    memory    = deque(maxlen=MEMORY_SIZE)

    epsilon         = EPSILON_START
    episode_rewards = []

    for episode in range(EPISODES):
        state        = env.reset()
        total_reward = 0.0

        # [v5] NLOS 구간 설정
        # NLOS_EPISODE_RATIO 확률로 이번 에피소드에 NLOS 구간 포함
        use_nlos    = random.random() < NLOS_EPISODE_RATIO
        nlos_start  = random.randint(10, STEPS_PER_EP - NLOS_STEPS - 5) if use_nlos else -1
        nlos_end    = nlos_start + NLOS_STEPS if use_nlos else -1

        for step in range(STEPS_PER_EP):

            # [v5] NLOS 급변 여부 결정
            is_nlos = use_nlos and (nlos_start <= step < nlos_end)

            # ε-greedy 행동 선택
            if random.random() < epsilon:
                action = random.randrange(env.action_dim)
            else:
                with torch.no_grad():
                    q = policy_net(
                        torch.FloatTensor(state).unsqueeze(0).to(device))
                action = q.argmax().item()

            # [v5] NLOS 이벤트를 step()에 전달하여 학습 중에도 경험
            next_state, reward, done, _ = env.step(action, nlos_event=is_nlos)
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

                # Gradient clipping: 학습 안정성 향상
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)

                optimizer.step()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)

        # Target Network 동기화
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            nlos_tag = " [NLOS 포함]" if use_nlos else ""
            print(f"  Episode {episode:4d}{nlos_tag} | "
                  f"Reward: {total_reward:8.2f} | ε: {epsilon:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # [v5] LR 스케줄러 스텝
        scheduler.step()

    torch.save(policy_net.state_dict(), "dqn_adaptive_cp.pth")
    np.save("train_rewards.npy", np.array(episode_rewards))
    print("\n[Train] Complete. Model saved → dqn_adaptive_cp.pth")


if __name__ == "__main__":
    train()
