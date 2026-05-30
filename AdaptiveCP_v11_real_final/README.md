# AdaptiveCP v11 — DQN 기반 적응형 CP 조절 알고리즘

> **논문:** "Impact of Beamforming on Delay Spread in Wideband Millimeter-wave Systems"  
> Akgun, Krunz, Manzi — ICNC 2020  
> **표준:** 3GPP TR 38.900 (채널 모델 파라미터)

---

## 1. 프로젝트 개요

고층 건물, 터널, 산간 지형 등 장애물이 많은 NLOS(Non-Line-of-Sight) 환경에서는 전파의 반사·굴절·회절로 인해 다중 경로 오류(Multipath Error)가 발생한다. 특히 UAV·자율주행차 등 고속 이동체에서는 채널의 DS(Delay Spread)가 동적으로 변화하는 시간 선택적 페이딩에 노출되어, 통신 단절 시 제어 상실로 이어지는 2차 사고 위험이 존재한다.

기존 OFDM 시스템은 고정 길이의 CP(Cyclic Prefix)를 사용한다. CP가 DS보다 짧으면 ISI(Inter-Symbol Interference)가 발생해 BER이 급증하고, 반대로 CP를 과도하게 길게 설정하면 유효 데이터 전송 구간이 줄어 주파수 및 전력 효율이 저하된다.

본 연구는 DQN(Deep Q-Network) 기반 강화학습을 활용하여 채널의 RMS Delay Spread 추정치에 따라 CP 길이를 실시간으로 적응 조절하는 알고리즘을 구현한다. 레퍼런스 논문의 빔포밍 채널 모델(키홀 안테나 모델)과 3GPP TR 38.900 UMa DS 통계를 시뮬레이션 환경으로 삼아, ISI를 방지하면서 Spectral Efficiency(SE)를 극대화하는 것이 목표이다.

---

## 2. v11 주요 변경 사항 (v10 대비)

### 2.1 신규 평가 지표 추가 (env.py)

SINR과 BER을 No CP / Theoretical / Fixed CP / Adaptive CP 4가지 케이스 모두에 대해 매 스텝 계산하여 `info` dict에 포함한다.

**SINR (Signal-to-Interference-plus-Noise Ratio)**

지수 PDP 가정 하에 CP 윈도우 밖으로 유출되는 ISI 에너지 분율을 계산한다.

```
ISI_fraction = exp(−CP / τ̂)
  · CP = 0    → ISI_fraction = 1.0  (모든 멀티패스 에너지가 ISI)
  · CP ≫ τ̂  → ISI_fraction → 0   (ISI 거의 없음)

SINR = SNR_linear / (1 + SNR_linear × ISI_fraction)
```

**BER (Bit Error Rate) — BPSK 근사**

```
BER = 0.5 × erfc(√SINR)
```

### 2.2 No CP 케이스 시각화 추가 (SE 차트 05, 08, 09)

레퍼런스 논문의 핵심 포인트 — *"CP를 사용하지 않더라도 고정 CP 대비 SE가 높은 경우가 있다"* — 를 명시적으로 시각화한다.

```
SE_no_cp  = (T_s / T_s) × log₂(1+SNR) × 0.5
          =  log₂(1+SNR) × 0.5
            ↑ 오버헤드 없음     ↑ ISI 패널티

SE_fixed  = (T_s / (T_s + 75 ns)) × log₂(1+SNR)
           ≈  0.118 × log₂(1+SNR)
              ↑ 오버헤드 손실 큼
```

`T_s = 10 ns` 기준, CP = 75 ns이면 오버헤드 손실이 88.2%이므로 ISI 패널티(50%)가 있는 No CP보다 SE가 낮아진다.

### 2.3 안테나 배열별 개별 저장

테스트를 한 번 실행하면 M = 16, 64, 256 세 케이스를 순서대로 수행하고, 결과를 각각 별도 서브폴더에 저장한다.

```
charts/
├── M=16/   — M=16 전용 결과 13개 차트
├── M=64/   — M=64 전용 결과 13개 차트
└── M=256/  — M=256 전용 결과 13개 차트
```

### 2.4 파라미터 비교표 추가 (chart_12)

고정 CP 대비 Adaptive CP의 평균 지표 변화를 한 장의 표로 정리한다.  
행: σ_DS, SNR, CP 길이, CP/σ_DS, SE(전체/Normal/NLOS), SINR, BER, Power Eff., ISI 위반 / 열: No CP · Theoretical · Fixed CP · Adaptive DQN · Δ vs Fixed

### 2.5 차트 색상 전체 통일

모든 차트에서 CP 케이스별 색상을 동일하게 고정한다.

| 색상 | CP 케이스 |
|---|---|
| 🟢 초록 `#2ca02c` | Adaptive DQN |
| 🔴 빨강 `#d62728` | Fixed CP |
| ⚫ 진회색 `#636363` | Theoretical (3·σ_DS) |
| 🔵 파랑 `#1f77b4` | No CP (CP = 0) |

---

## 3. 파일 구조

```
AdaptiveCP_v11/
├── env.py       — 강화학습 환경 (채널 모델, PDP, DS, SINR, BER 계산)
├── DQN.py       — 신경망 구조 (3층 FC, 입력 정규화)
├── train.py     — 학습 루프 (ε-greedy, Experience Replay, Target Network)
├── test.py      — M=16/64/256 평가 및 차트 저장 (13개 × 3 = 39개)
├── dqn_adaptive_cp.pth    (train 후 생성 / v10 모델 재사용 가능)
├── train_rewards.npy      (train 후 생성 / v10 데이터 재사용 가능)
└── charts/
    ├── M=16/
    ├── M=64/
    └── M=256/
```

### 실행

```powershell
# 학습 (이미 v10 모델이 있으면 생략 가능)
py -3.11 train.py

# 평가 — M=16, 64, 256 자동 3회 실행
py -3.11 test.py
```

> v11 `test.py`는 `FIXED_M` 설정이 필요 없다. `M_VALUES = [16, 64, 256]` 상수가 세 케이스를 자동으로 실행한다.

### 출력 차트 목록 (M별 동일)

| 파일 | 내용 |
|---|---|
| `01_training_reward.png` | 학습 리워드 수렴 곡선 |
| `02_step_reward.png` | 스텝별 리워드 |
| `03_sigma_ds.png` | RMS Delay Spread 추이 |
| `04_cp_comparison.png` | CP 길이 비교 (Adaptive / Fixed / Theo.) |
| `05_se_comparison.png` | SE 비교 **[v11: No CP 추가]** |
| `06_power_efficiency.png` | 전력 효율 비교 |
| `07_cp_sigma_ratio.png` | CP / σ_DS 비율 (ISI 안전 마진) |
| `08_avg_se_summary.png` | 평균 SE 바 차트 **[v11: No CP 추가]** |
| `09_nlos_se_comparison.png` | Normal/NLOS 구간 SE 비교 **[v11: No CP 추가]** |
| `10_ber_comparison.png` | BER 비교, log 스케일 **[v11 신규]** |
| `11_sinr_comparison.png` | SINR 비교 **[v11 신규]** |
| `12_param_table.png` | 평균 파라미터 비교표 **[v11 신규]** |
| `summary.png` | 핵심 4차트 요약 (학습/SE/평균SE/SINR) |

---

## 4. 채널 모델 및 수식 (논문 기반)

본 프로젝트는 레퍼런스 논문의 수식과 파라미터를 변형 없이 그대로 사용한다.

### 4.1 Power Delay Profile (논문 Eq. 5)

```
Pc(τ) = (1 / 2τ̂) · exp(−2τ / τ̂) · [G(θ⁻(τ)) + G(θ⁺(τ))]

θ⁻(τ) = θ_c − κ · √(τ / τ̂)
θ⁺(τ) = θ_c + κ · √(τ / τ̂)
```

- `τ̂` : 평균 초과 지연 (mean excess delay)
- `θ_c` : 클러스터 중심 도래각
- `κ = 5.0°` : 각도 확산 계수 (논문 고정값)

### 4.2 RMS Delay Spread 계산 (논문 Eq. 6–7)

```
μ_DS  = ∫ τ · Pc(τ) dτ  /  ∫ Pc(τ) dτ
σ_DS  = √[ ∫ τ² · Pc(τ) dτ / ∫ Pc(τ) dτ  −  μ_DS² ]
```

수치 적분(`scipy.integrate.quad`)으로 매 스텝 계산하며, 채널 상태 변화가 즉시 반영된다.

### 4.3 UPA 빔포밍 모델 (논문 Eq. 3–4)

```
ωb = 1.772 / (√M · cos θb)    — 빔폭
G_m = π · M                   — 메인로브 이득
g_m = G_m × 10^(−13.26/10)   — 사이드로브 이득
```

안테나 배열 M ∈ {16, 64, 256}이 빔폭과 이득에 직접 영향을 준다.  
M이 클수록 빔이 좁아져 LOS 이득이 높아지지만, 빔 미정렬(NLOS) 시 SE 손실이 커진다.

### 4.4 Spectral Efficiency (논문 Eq. 8)

```
SE = (T_s / (T_s + T_CP)) · log₂(1 + SNR)
```

CP가 σ_DS보다 짧으면(CP < σ_DS) ISI 발생으로 SE를 50% 패널티 처리한다.

**No CP 케이스 (v11 추가):**

```
SE_no_cp = 1.0 × log₂(1 + SNR) × 0.5     [CP = 0, ISI 패널티 적용]
```

CP 오버헤드가 없으므로 효율 계수는 1.0이지만, CP = 0 < σ_DS 이므로 ISI 패널티가 적용된다.  
이 모델에서 T_s = 10 ns이고 Fixed CP = 75 ns이면, `T_s / (T_s + 75 ns) ≈ 0.118`로 오버헤드 손실이 매우 크기 때문에 `0.5 × log₂(1+SNR) > 0.118 × log₂(1+SNR)` 관계가 성립한다.  
이것이 **"No CP SE > Fixed CP SE"** 라는 논문의 핵심 포인트를 수식으로 뒷받침한다.

### 4.5 논문 기준 Theoretical CP

```
T_CP = ρ · σ_DS    (ρ = 3.0, 논문 reference [16])
Fixed CP = τ̂_ref × ρ = 25 ns × 3.0 = 75 ns
```

---

## 5. 3GPP TR 38.900 채널 파라미터

채널 초기화 시 DS 통계는 3GPP TR 38.900 Table 7.5-6 UMa 모델을 따른다.

| 시나리오 | μ (log₁₀[DS/s]) | σ (log₁₀[DS/s]) | SNR 범위 | 빔 정렬 |
|---|---|---|---|---|
| LOS  | −7.03 | 0.66 | 10 ~ 25 dB | θ_b = θ_c, Δθ = 0° |
| NLOS | −6.44 | 0.39 |  3 ~ 12 dB | θ_b ∈ [30°,70°], θ_c ∈ [0°,20°] |

`τ̂`는 3GPP DS 샘플로부터 다음과 같이 환산한다.

```
τ̂ = DS_3gpp × (τ̂_ref / DS_LOS_median)
τ̂_ref = 25 ns,  DS_LOS_median = 10^(−7.03) s
```

---

## 6. 강화학습 구조

### 6.1 State 벡터 (6차원)

| 인덱스 | 변수 | 물리적 의미 | 정규화 기준 |
|---|---|---|---|
| 0 | σ_DS (ns) | RMS Delay Spread | 200 ns |
| 1 | SNR (dB) | 수신 SNR | 30 dB |
| 2 | Δθ (°) | 빔 정렬 오차 | 45° |
| 3 | M | 안테나 배열 수 | 256 |
| 4 | T_CP (ns) | 현재 CP 길이 | 600 ns |
| 5 | is_nlos | NLOS 플래그 (0 / 1) | 1 |

모든 입력은 DQN 내부에서 정규화 기준값으로 나눠 [0, 1] 범위로 스케일링된다.

### 6.2 Action 공간 (3가지)

| Action | 동작 |
|---|---|
| 0 | CP 감소 (Decrease CP) |
| 1 | CP 유지 (Maintain CP) |
| 2 | CP 증가 (Increase CP) |

스텝 크기는 현재 CP / σ_DS 비율에 따라 동적으로 결정된다.

```
CP/σ_DS < 0.8  →  step = σ_DS × 0.5   (빠른 증가 필요)
CP/σ_DS < 1.5  →  step = σ_DS × 0.2   (미세 조정)
CP/σ_DS < 5.0  →  step = σ_DS × 0.3
CP/σ_DS ≥ 5.0  →  step = σ_DS × 0.5   (과도 CP 빠른 감소)
```

CP 하한은 항상 `σ_DS × 1.1` 이상으로 클램핑하여 ISI를 원천 차단한다.

### 6.3 Reward 함수

```python
reward = (SE_adaptive − SE_fixed) × 20.0     # Fixed CP 대비 SE 개선이 기본 보상

if 1.1·σ_DS ≤ CP ≤ 2.0·σ_DS:
    reward += 3.0                             # 효율적 CP 범위 보너스

if CP > 3.0·σ_DS:
    overrun = CP / (3.0·σ_DS) − 1.0
    reward −= overrun × 5.0                  # 논문 기준값 초과 패널티

reward = clip(reward, −20.0, 20.0)
```

설계 의도: Fixed CP 대비 SE를 높이는 방향을 학습시키되, CP가 논문 기준값(3σ)을 초과해 불필요하게 길어지는 것을 억제한다. 1.1σ ~ 2.0σ 구간에 안착하면 보너스를 부여하여 ISI 안전성과 효율성을 동시에 달성하도록 유도한다.

### 6.4 DQN 신경망 구조

```
입력 (6차원, 정규화)
    │
FC(6 → 128) + ReLU
    │
FC(128 → 128) + ReLU
    │
FC(128 → 3)  — 각 Action의 Q-value 출력
```

### 6.5 학습 하이퍼파라미터

| 파라미터 | 값 |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 (CosineAnnealingLR, η_min = 1e-4) |
| Discount Factor γ | 0.99 |
| ε-greedy | 1.0 → 0.01 (decay = 0.995) |
| Batch Size | 64 |
| Replay Memory | 20,000 |
| Episodes | 2,000 |
| Steps / Episode | 100 |
| Target Network 갱신 주기 | 10 에피소드 |
| Gradient Clipping | max_norm = 1.0 |
| NLOS 에피소드 비율 | 40% |
| NLOS 지속 스텝 수 | 10 스텝 |

---

## 7. 최종 평가 결과

> 테스트 조건: 30 스텝, NLOS 구간 Step 10–18 (9스텝), M = 16 / 64 / 256 개별 실행

### 7.1 테스트 프로토콜 — 차트 x축 `step`의 의미

모든 차트의 x축 `step`은 **서로 독립적인 재실행이 아니라, 한 번 초기화한 채널이 시간에 따라 연속적으로 변해가는 단일 에피소드의 시간 인덱스**이다. `test.py`의 `run_single()`은 M값마다 `env.reset()`을 한 번만 호출(채널 1회 초기화)한 뒤, 30 스텝 루프를 돌며 매 스텝 `env.step()`으로 채널 상태를 갱신한다. 루프 내부에 reset이 없으므로 30 스텝 전체가 하나의 에피소드에 속한다. 즉 "같은 환경을 반복 측정"한 것도, "매 스텝 완전히 새로운 랜덤 환경"도 아니다.

채널은 매 스텝 두 가지 방식으로 변한다.

- **LOS 구간 (Step 1–9, 19–30) — 점진 변화(시간 상관)**
  - `τ̂ ← 0.6·τ̂ + 0.4·(새 LOS 샘플)` — 지수 평활로 서서히 드리프트(점프 아님)
  - `SNR ← SNR + Uniform(−1, +1) dB`, [5, 30] dB 클램프 — 랜덤 워크
  - 갱신된 τ̂로 σ_DS를 매 스텝 재적분
- **NLOS 구간 (Step 10–18) — 급변(이벤트 주입)**
  - `nlos_event=True`가 주입되어 채널이 급격히 NLOS로 전환(SNR 3–12 dB, 빔 미정렬, τ̂를 NLOS 통계에서 재샘플링)
  - 차트의 음영 처리된 **NLOS Zone**이 이 구간이다

```
Step:  1 ─ 2 ─ … ─ 9 │ 10 ─ … ─ 18 │ 19 ─ … ─ 30
       └─── LOS ─────┘ └── NLOS ───┘ └─── LOS ─────┘
         점진 드리프트     급변(이벤트)     점진 드리프트
```

따라서 각 차트는 **이동체가 겪는 시간 선택적 페이딩 채널의 30 스텝 시계열 한 트랙**이며, 중간에 NLOS 단절 이벤트를 의도적으로 주입한 시나리오이다. NLOS 진입으로 σ_DS가 커지면 Adaptive CP가 따라 올라갔다가 LOS 복귀 시 다시 내려오는 거동이 곧 적응형 알고리즘의 핵심을 보여준다.

> **재현성 참고:** 코드에 난수 시드가 고정되어 있지 않아 `test.py`를 재실행하면 매번 다른 채널 트랙이 생성된다. 아래 수치는 특정 1회 실행 결과이며, 에이전트의 행동 정책은 결정적(argmax)이지만 채널 샘플링이 확률적이므로 실행마다 값이 소폭 달라진다. M = 16 / 64 / 256은 각각 독립적인 reset + 30 스텝 에피소드로 따로 실행되어 별도 폴더에 저장된다.

### 7.2 평균 Spectral Efficiency (bps/Hz)

| M | No CP | Theoretical | Fixed CP | **Adaptive (DQN)** | Adaptive vs Fixed |
|---|---|---|---|---|---|
| 16 | 1.3636 | 0.3583 | 0.3130 | **0.6562** | **+109.6%** |
| 64 | 1.2263 | 0.2650 | 0.2885 | **0.5171** | **+79.2%** |
| 256 | 1.3904 | 0.4664 | 0.3272 | **0.7599** | **+132.3%** |

### 7.3 주요 지표 상세 (M = 64 기준)

| 지표 | No CP | Theoretical | Fixed CP | Adaptive (DQN) | Δ vs Fixed |
|---|---|---|---|---|---|
| σ_DS mean (ns) | — | 33.42 | 33.42 | 33.42 | — |
| SNR mean (dB) | — | 6.45 | 6.45 | 6.45 | — |
| CP mean (ns) | 0 | 100.25 | 75.00 | 40.95 | −34.05 ns |
| CP / σ_DS mean | 0 | 3.00 | 2.24 | 1.33 | −0.915 |
| SE mean (bps/Hz) | 1.2263 | 0.2650 | 0.2885 | 0.5171 | +79.2% |
| SE Normal (bps/Hz) | 1.2004 | 0.3080 | 0.2824 | 0.5698 | +101.7% |
| SE NLOS (bps/Hz) | 1.2869 | 0.1648 | 0.3028 | 0.3940 | +30.1% |
| SINR mean (dB) | −0.93 | 3.40 | 2.90 | 1.21 | −1.69 dB |
| BER mean | 1.02e-01 | 1.94e-02 | 2.87e-02 | 5.26e-02 | ×1.8 worse |
| Power Eff. mean (%) | 100.0 | 10.99 | 11.76 | 21.20 | +9.43 pp |
| ISI Violations | 30/30 | 0 | 0 | 0/30 | — |

### 7.4 핵심 관찰

**① No CP SE > Fixed CP SE — 논문 핵심 포인트 검증**  
모든 M값, 모든 스텝에서 No CP SE가 Fixed CP SE를 상회하였다. Fixed CP(75 ns)는 실제 σ_DS(~27–33 ns) 대비 약 2.2–2.8배 과도하여, CP 오버헤드 손실(오버헤드 계수 ≈ 0.118)이 No CP의 ISI 패널티(×0.5)보다 크기 때문이다. 이는 논문이 Adaptive CP 필요성을 주장하는 근거를 수치로 확인한 것이다.

**② Adaptive CP의 트레이드오프 — SE ↑, SINR/BER은 절충 (M = 64 기준)**  
Adaptive DQN은 CP를 평균 40.95 ns(Fixed 75 ns 대비 −45%)로 운용하여 SE를 +79.2% 향상시키고 전력 효율도 +9.43 pp 개선하였다. 반면 CP가 짧아진 만큼 ISI 에너지 분율이 증가하여 SINR이 −1.69 dB 낮아지고 BER은 약 1.8배 높아진다. 이 트레이드오프는 SE 최대화를 목표로 설계된 보상 함수의 의도된 결과이다.

**③ ISI 위반 0건 (Adaptive)**  
CP 하한 클램프(σ_DS × 1.1)가 NLOS 급변 구간에서도 ISI를 완전히 차단하였다(전 M값에서 0/30).

**④ M별 Adaptive SE 향상폭**  
Adaptive SE 향상폭은 M = 256에서 +132.3%로 가장 크고, M = 16에서 +109.6%, M = 64에서 +79.2% 순으로, M에 대해 단조 증가하지는 않는다. M = 256은 빔이 가장 좁아 포착되는 멀티패스 성분이 줄고 σ_DS가 가장 낮으며(27.22 ns), 이에 따라 Adaptive CP도 가장 짧게 유지되어(34.84 ns, CP/σ_DS 1.48) SE 향상폭과 전력 효율 개선폭(+13.96 pp)이 모두 가장 컸다. M = 64는 평균 SNR(6.45 dB)이 세 케이스 중 가장 낮고 σ_DS(33.42 ns)가 가장 커서 향상폭이 상대적으로 작게 나타난다. 즉 향상폭은 빔폭·포착 멀티패스·SNR이 함께 작용한 결과이다.

---

## 8. 결과 해석 시 주의 사항

### 8.1 No CP ISI Violations = 30/30 의 의미

파라미터 비교표에서 No CP의 ISI Violations가 항상 **30/30** 으로 표시된다. 이는 다음 이유에서 수학적으로 정확한 값이다.

- 이 시뮬레이션의 ISI Violation 정의: **CP < σ_DS** (즉 CP/σ_DS < 1.0)
- No CP이면 CP = 0, σ_DS는 항상 > 0 → CP/σ_DS = 0 < 1.0 → 항상 위반

그러나 이것이 "No CP이면 반드시 통신이 불가능하다"를 의미하지는 않는다.  
**ISI Violation은 권장 CP 기준(CP ≥ σ_DS) 미달 여부이지, 채널의 물리적 ISI 발생 여부가 아니다.**

실제 ISI 심각도는 σ_DS / T_s 비율에 의존한다.

| 조건 | 의미 |
|---|---|
| σ_DS ≪ T_s | No CP여도 ISI 영향 미미 |
| σ_DS ≈ T_s | No CP시 경미한 ISI |
| σ_DS > T_s | No CP시 심각한 ISI **(본 시뮬레이션 해당: σ_DS ≈ 27–33 ns > T_s = 10 ns)** |

이 시뮬레이션에서는 σ_DS 평균 ≈ 27–33 ns(M=16/64/256), T_s = 1/100 MHz = 10 ns로 σ_DS > T_s이므로, No CP시 실제로도 유의미한 ISI가 발생하는 상황이다. BER 차트(10번)에서 No CP의 BER이 ~10⁻¹ 수준으로 Fixed CP(평균 ~2–3×10⁻²)보다 평균 약 4배(구간에 따라 최대 10배 가까이) 높게 나오는 것이 이를 확인시켜 준다.

결론: 30/30은 코드 버그가 아니라, 이 시뮬레이션 파라미터 조건(σ_DS > T_s) 하에서 물리적으로도 타당한 결과이다.

### 8.2 SINR 역전 현상 (Fixed > Adaptive)

Fixed CP(75 ns)의 SINR이 Adaptive CP(~35–43 ns)보다 높게 나오는 것은 정상적인 결과이다.  
Fixed CP가 σ_DS의 약 2.2–2.8배로 유지되어 ISI가 거의 없는 반면, Adaptive CP는 의도적으로 CP를 줄여 SE를 높이는 대신 일부 ISI를 허용하기 때문이다.  
이 역전이 바로 **"SE ↑ vs SINR/BER ↓"** 트레이드오프의 시각적 표현이다.

---

## 9. 의존성

```
Python   >= 3.11
torch
numpy
scipy
matplotlib
seaborn
```
