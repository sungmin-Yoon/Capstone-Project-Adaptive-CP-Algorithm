import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    3층 완전연결 신경망 (Fully-Connected Network)

    v5: 입력 정규화, 은닉층 64→128
    v8: STATE_NORM 5차원 → 6차원 (is_nlos 플래그 추가)
        입력 차원: state_dim=6
    v9: 변경 없음 (네트워크 구조 유지)
    """

    # 6차원 정규화 기준값
    # [σ_DS(ns), SNR(dB), Δθ(°), M, CP(ns), is_nlos]
    STATE_NORM = torch.tensor(
        [200.0, 30.0, 45.0, 256.0, 600.0, 1.0],
        dtype=torch.float32
    )

    def __init__(self, state_dim: int, action_dim: int):
        super(DQN, self).__init__()
        self.register_buffer("state_norm", self.STATE_NORM)

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.state_norm.to(x.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
