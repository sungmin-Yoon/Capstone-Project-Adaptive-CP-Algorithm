import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    """
    3층 완전연결 신경망 (Fully-Connected Network)

    ── v5 변경 사항 ──────────────────────────────────────────────────────────
    입력 정규화 추가:
      State 값의 스케일이 제각각 (σ_DS: 수~200ns, M: 16~256, CP: 수~600ns)
      이로 인해 특정 입력이 gradient를 지배하여 학습 불안정.
      STATE_NORM 기준값으로 [0, 1] 범위로 정규화 후 네트워크에 입력.

    State 정규화 기준 (AdaptiveCPEnv.STATE_NORM):
      [σ_DS/200, SNR/30, Δθ/45, M/256, CP/600]
    """

    # env.py의 STATE_NORM과 동일한 값
    STATE_NORM = torch.tensor(
        [200.0, 30.0, 45.0, 256.0, 600.0],
        dtype=torch.float32
    )

    def __init__(self, state_dim: int, action_dim: int):
        super(DQN, self).__init__()
        self.register_buffer("state_norm", self.STATE_NORM)

        # 은닉층 크기를 128로 확대 (정규화 후 표현력 향상)
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력 정규화: 각 State 차원을 [0, 1] 범위로 스케일링
        x = x / self.state_norm.to(x.device)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
