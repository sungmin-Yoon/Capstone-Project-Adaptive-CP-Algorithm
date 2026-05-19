import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from env import AdaptiveCPEnv
from DQN import DQN

def test():
    env = AdaptiveCPEnv()
    device = torch.device("cpu") # GPU 환경이라면 "cuda" 사용
    
    model = DQN(env.state_dim, env.action_dim).to(device)
    model.load_state_dict(torch.load("dqn_adaptive_cp.pth"))
    model.eval()
    
    print("--- Testing Adaptive CP Algorithm ---")
    state = env.reset()
    
    action_map = {0: "Decrease CP", 1: "Maintain CP", 2: "Increase CP"}
    
    # 그래프를 위한 데이터 수집 리스트
    step_rewards = []
    theo_se_list = []
    adapt_se_list = []
    theo_power_eff_list = [] # 이론적 전력 효율 리스트 추가
    power_eff_list = []
    
    num_steps = 20
    
    for step in range(num_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        
        action = q_values.argmax().item()
        next_state, reward, done, info = env.step(action)
        
        # 데이터 수집
        step_rewards.append(reward)
        theo_se_list.append(info['theo_se'])
        adapt_se_list.append(info['adaptive_se'])
        theo_power_eff_list.append(info['theo_power_eff']) # 데이터 저장
        power_eff_list.append(info['power_eff'])
        
        print(f"Step {step+1} | DS: {state[0]:.2f}ns | SNR: {state[1]:.1f}dB | M: {state[3]}")
        print(f"  [이론적 기준] CP: {info['theo_cp']*1e9:.2f}ns | SE: {info['theo_se']:.4f} bps/Hz | Eff: {info['theo_power_eff']:.2f}%")
        print(f"  [적응형 판단] Action: {action_map[action]} | CP: {info['adaptive_cp']*1e9:.2f}ns | SE: {info['adaptive_se']:.4f} bps/Hz | Eff: {info['power_eff']:.2f}%")
        print(f"  [효율 지표] RSSI: {info['rssi']:.2f} | Reward: {reward:.4f}")
        print("-" * 65)
        
        state = next_state

    # ---------------------------------------------------------
    # 통합 대시보드 시각화 (2x2 Subplots)
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Adaptive CP Algorithm Performance Dashboard", fontsize=16, fontweight='bold')
    
    # 1. 학습 보상 그래프
    if os.path.exists("train_rewards.npy"):
        train_rewards = np.load("train_rewards.npy")
        axs[0, 0].plot(train_rewards, color='blue', alpha=0.7)
        axs[0, 0].set_title("Training Phase: Total Reward per Episode")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Total Reward")
        axs[0, 0].grid(True, linestyle='--', alpha=0.5)
    else:
        axs[0, 0].text(0.5, 0.5, "No Train Data Found", ha='center', va='center')
        
    # 2. 테스트 스텝별 보상 그래프
    axs[0, 1].plot(range(1, num_steps + 1), step_rewards, marker='o', color='orange', linewidth=2)
    axs[0, 1].set_title("Test Phase: Step-by-Step Reward")
    axs[0, 1].set_xlabel("Step")
    axs[0, 1].set_ylabel("Reward")
    axs[0, 1].set_xticks(range(1, num_steps + 1))
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    # 3. Spectral Efficiency (SE) 비교
    axs[1, 0].plot(range(1, num_steps + 1), theo_se_list, marker='s', linestyle='--', color='gray', label='Theoretical SE')
    axs[1, 0].plot(range(1, num_steps + 1), adapt_se_list, marker='^', color='green', label='Adaptive SE (DQN)')
    axs[1, 0].set_title("Spectral Efficiency (SE) Comparison")
    axs[1, 0].set_xlabel("Step")
    axs[1, 0].set_ylabel("SE (bps/Hz)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    
    # 4. 전력 효율 (Power Efficiency) 비교 - [수정된 부분]
    axs[1, 1].plot(range(1, num_steps + 1), theo_power_eff_list, marker='s', linestyle='--', color='gray', label='Theoretical Power Eff')
    axs[1, 1].plot(range(1, num_steps + 1), power_eff_list, marker='d', color='purple', linewidth=2, label='Adaptive Power Eff (DQN)')
    axs[1, 1].set_title("Power Efficiency Comparison")
    axs[1, 1].set_xlabel("Step")
    axs[1, 1].set_ylabel("Efficiency (%)")
    axs[1, 1].set_ylim(0, max(max(power_eff_list), max(theo_power_eff_list)) + 15) # 최대치에 맞춰 여유 공간 확보
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    test()