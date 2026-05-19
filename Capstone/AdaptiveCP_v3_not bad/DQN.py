import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    3층 완전연결 신경망 (Fully-Connected Network)
    - 입력: 5차원 상태 벡터 [σ_DS, SNR, Δθ, M, CP]
    - 은닉층: 64 뉴런 × 2, ReLU 활성화
    - 출력: 3개 행동(감소/유지/증가)에 대한 Q값
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
