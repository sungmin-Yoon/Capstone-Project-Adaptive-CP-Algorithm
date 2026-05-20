import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    3층 완전연결 신경망 (Fully-Connected Network)

    v5부터 적용:
    - 입력 정규화: STATE_NORM으로 [0,1] 범위 스케일링
    - 은닉층 64 → 128 확대

    v7: 변경 없음 (구조 동일)
    """

    STATE_NORM = torch.tensor(
        [200.0, 30.0, 45.0, 256.0, 600.0],
        dtype=torch.float32
    )

    def __init__(self, state_dim: int, action_dim: int):
        super(DQN, self).__init__()
        self.register_buffer("state_norm", self.STATE_NORM)

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력 정규화
        x = x / self.state_norm.to(x.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
