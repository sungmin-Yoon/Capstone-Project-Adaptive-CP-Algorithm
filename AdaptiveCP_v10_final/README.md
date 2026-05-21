# AdaptiveCP — DQN 기반 적응형 CP 조절 알고리즘

> **논문 기반:** "Impact of Beamforming on Delay Spread in Wideband Millimeter-wave Systems"  
> Akgun, Krunz, Manzi — ICNC 2020

---

## 프로젝트 개요

NLOS 환경에서 고속 이동체의 통신 연속성 유지를 위해, DQN 기반 강화학습을 활용하여  
채널의 DS(Delay Spread) 추정치에 따라 CP(Cyclic Prefix) 길이를 동적으로 조절하는 알고리즘 연구.

---

## 파일 구조

```
AdaptiveCP_v10/
├── env.py          # 강화학습 환경 (채널 모델, PDP, DS 계산, 보상 함수)
├── DQN.py          # 신경망 구조 (3층 FC, 입력 정규화)
├── train.py        # 학습 루프 (ε-greedy, Experience Replay, Target Network)
├── test.py         # 평가 + 개별 차트 저장 (Option D 시각화)
├── dqn_adaptive_cp.pth      (train 후 생성)
├── train_rewards.npy        (train 후 생성)
└── charts/                  (test 후 생성)
    ├── 01_training_reward.png
    ├── 02_step_reward.png
    ├── 03_sigma_ds.png
    ├── 04_cp_comparison.png
    ├── 05_se_comparison.png
    ├── 06_power_efficiency.png
    ├── 07_cp_sigma_ratio.png
    ├── 08_avg_se_summary.png
    ├── 09_nlos_se_comparison.png
    └── summary_top4.png
```

---

## 실행 방법

```powershell
py -3.11 train.py
py -3.11 test.py
```

---

## 안테나 배열 선택 (v10 신규)

`train.py` 및 `test.py` 상단의 `FIXED_M` 상수로 안테나 배열을 선택합니다.

```python
FIXED_M = None   # None = 랜덤 (M ∈ {16, 64, 256})
                 # 16   = M=16  안테나 배열로 고정
                 # 64   = M=64  안테나 배열로 고정
                 # 256  = M=256 안테나 배열로 고정
```

| FIXED_M 설정 | 동작 | 차트 표기 |
|---|---|---|
| `None` | 매 reset() 마다 랜덤 선택 | "안테나 배열: Random 선택 결과 M = X" |
| `16` | M=16 고정 | "안테나 배열: M = 16 (고정)" |
| `64` | M=64 고정 | "안테나 배열: M = 64 (고정)" |
| `256` | M=256 고정 | "안테나 배열: M = 256 (고정)" |

> train.py 와 test.py 의 FIXED_M 값을 일치시켜야 정확한 평가가 됩니다.

---

## 시각화 출력 (Option D)

`test.py` 실행 시 `charts/` 폴더에 10개 파일 자동 저장.

| 파일 | 내용 | 용도 |
|---|---|---|
| `01_training_reward.png` | 학습 보상 수렴 추이 | 학습 안정성 확인 |
| `02_step_reward.png` | 스텝별 보상 | NLOS 대응 확인 |
| `03_sigma_ds.png` | RMS Delay Spread 변화 | 채널 변화 시각화 |
| `04_cp_comparison.png` | CP 길이 3-way 비교 | CP 조절 동작 확인 |
| `05_se_comparison.png` | SE 3-way 비교 | **핵심 성능 차트** |
| `06_power_efficiency.png` | 전력 효율 3-way 비교 | 에너지 효율 확인 |
| `07_cp_sigma_ratio.png` | CP / σ_DS 비율 | ISI 안전성 확인 |
| `08_avg_se_summary.png` | 평균 SE 바 차트 + 개선율 | **보고서 핵심 차트** |
| `09_nlos_se_comparison.png` | 정상/NLOS 구간 SE | **보고서 핵심 차트** |
| `summary_top4.png` | 핵심 4차트 2×2 요약 | **발표 슬라이드용** |

- 해상도: 200 dpi, 개별 차트 10×6 inch, 요약 16×10 inch
- 모든 차트 하단 우측에 안테나 배열 정보 표기
- 한글 폰트: OS 감지 자동 설정 (Windows: Malgun Gothic)

---

## 버전 히스토리 요약

| 버전 | 핵심 변경 | Adaptive SE |
|---|---|---|
| v1~v2 | 프로토타입, 수식 재구현 | — |
| v3 | NLOS 모델링 최초 성공 | 0.956 |
| v4 | 3GPP TR 38.900 통합 | 0.900 |
| v5 | DQN 정규화, NLOS 타이밍 수정 | 0.438 |
| v6 | 보상 함수 재설계 (실패) | 0.263 |
| v7 | step_size 수정, 보상 기준 변경 | 1.146 |
| v8 | ISI 차단, is_nlos state, 논문 준수 복원 | 0.510 |
| v9 | NLOS 타이밍 충돌 해소 (최종 알고리즘) | **1.076** |
| **v10** | 안테나 선택, 한글 폰트, 개별 차트 저장 | 1.076 (동일) |

> v8/v9 의 Adaptive SE 차이는 theo_cp 기준 재정의(σ×1.1 → σ×3.0)에 따른 것으로,  
> 알고리즘 성능 자체는 v9 기준이 올바른 논문 준수 측정값입니다.

---

## v10 변경 사항

### [변경 1] 안테나 배열 선택 기능 (env.py + train.py + test.py)

```python
# env.py
def __init__(self, fixed_m=None):
    self.fixed_m = fixed_m

def reset(self):
    if self.fixed_m is not None:
        self.M = self.fixed_m        # 고정
    else:
        self.M = random.choice(self.M_options)  # 랜덤

# train.py / test.py 상단
FIXED_M = None  # 여기만 변경하면 됩니다
env = AdaptiveCPEnv(fixed_m=FIXED_M)
```

### [변경 2] 한글 폰트 자동 설정 (test.py)

```python
def setup_font():
    candidates = {
        "Windows": ["Malgun Gothic", "NanumGothic", ...],
        "Darwin":  ["AppleGothic", ...],
        "Linux":   ["NanumGothic", ...],
    }
    # OS 감지 후 설치된 폰트 자동 적용
    matplotlib.rcParams["axes.unicode_minus"] = False
```

### [변경 3] Option D 개별 차트 저장 (test.py)

- 9개 개별 차트 + 핵심 4차트 요약(summary_top4.png) 자동 저장
- 모든 차트 색상 통일: Adaptive=초록, Fixed=빨강, Theoretical=진회색
- `⑧ 평균 SE` 차트에 "+X% vs Fixed CP" callout 자동 계산 및 표기
- 모든 차트 하단 우측에 안테나 배열 정보 자동 삽입
