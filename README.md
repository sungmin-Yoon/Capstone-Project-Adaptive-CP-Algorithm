# Capstone-Project-Adaptive-CP-Algorithm
High-speed mobility in an NLOS environment To maintain communication continuity DS-based adaptive OFDM reception system Research

# NLOS 환경에서 고속 이동체의 통신 연속성 유지를 위한 DS 기반 적응형 OFDM 수신 시스템 연구

본 프로젝트는 고속 이동체가 직면하는 비가시선(NLOS, Non-Line-of-Sight) 다중 경로 환경에서, Delay Spread (DS) 증가에 따른 통신 성능 저하를 극복하기 위한 적응형 수신 시스템 설계 연구입니다. 

강화학습(Deep Q-Network, DQN)을 활용하여 실시간 채널 상태(DS, SNR 등)를 관측하고, 이에 맞춰 보호 구간(Cyclic Prefix, CP)의 길이와 수신 링크를 동적으로 최적화하는 알고리즘을 제안합니다.

---

## 1. 프로젝트 배경 및 문제 정의

* **다중 경로 페이딩과 ISI 발생:** 도심 환경 등에서 전파의 반사, 회절로 인해 다중 경로 컴포넌트(MPC)가 수신기에 도달하며, 이는 Delay Spread(DS)를 증가시킵니다. DS가 증가하면 심볼 간 간섭(ISI)이 발생하여 결과적으로 Bit Error Rate (BER) 증가 및 SINR 저하로 이어집니다.
* **고정형 CP의 한계:** 기존 OFDM 시스템은 ISI 완화를 위해 고정된 길이의 CP를 사용합니다. 하지만 CP가 DS보다 짧으면 ISI를 제거하지 못하고, 반대로 과도하게 길면 CP 오버헤드로 인해 유효 데이터 전송 효율이 급감하는 상충 관계(Trade-off)가 발생합니다.

<img width="699" height="554" alt="image" src="https://github.com/user-attachments/assets/440a0dc4-d364-4126-86b0-d46e1754145d" />
*(그림 1. 도심 환경에서의 Tx와 Rx 간 통신 경로 및 Multipath 현상)*

<br><br>

<img width="419" height="396" alt="image" src="https://github.com/user-attachments/assets/b56b0def-900f-4d10-b94d-f6c0c739b3fd" />
<br>*(그림 2. DS 증가에 따른 ISI 발생 개념도)*

## 2. 시스템 핵심 아키텍처

본 시스템은 다중 경로 환경 모델링, 수신 빔포밍, DS 실시간 계산, 그리고 DQN 기반 적응형 CP 제어 모듈로 구성됩니다.

### 2.1. Multipath Channel Modeling & Beamforming
* **채널 모델:** 3GPP UMa LOS 채널 모델을 기반으로 다중 경로 환경을 구축합니다.
* **UPA 기반 빔포밍:** 수신단에 Uniform Planar Array (UPA)를 적용하여 빔포밍을 수행합니다. 안테나 배열 크기와 빔 방향에 따라 수신되는 유효 경로 전력과 DS가 물리적으로 변화하는 특성을 반영합니다.

### 2.2. DQN 기반 적응형 CP 최적화 알고리즘
DQN 에이전트는 ISI 방지와 전송 효율 최대화 사이의 최적점을 실시간으로 학습합니다.
* **상태 (State):** DS (ns), SNR (dB), 빔 정렬 오차 (degree), 안테나 수 (M), 현재 CP 길이 (ns)의 5차원 벡터.
* **행동 (Action):** CP 길이를 5ns 단위로 감소, 유지, 증가 (3가지 이산 행동).
* **보상 (Reward):** Spectral Efficiency (SE) 개선분과 수신 신호 세기(RSSI)를 기본으로 하되, ISI 발생 시 강한 페널티(-100)를, 최적 구간(DS의 1.0~1.5배) 진입 시 보너스(+10)를, 과잉 CP 구간 시 오버헤드 페널티(-5)를 부여합니다.

<img width="1536" height="845" alt="image" src="https://github.com/user-attachments/assets/06c16b9c-169b-42c8-af1c-297dbc7584b2" />
*(그림 3. CP 길이에 따른 전송 효율 및 데이터 전송률 - CP 오버헤드로 인한 효율 감소 추이)*

## 3. 시뮬레이션 환경 및 주요 결과

본 연구는 MATLAB을 통한 이론적 수식 검증과 Python (PyTorch) 기반의 강화학습 모델 구현으로 나뉘어 진행되었습니다.

### 3.1. 빔 방향 및 오정렬에 따른 DS 특성 (MATLAB)
* 안테나 배열 크기가 커질수록 빔폭이 좁아져 특정 방향의 경로만 수신하므로 RMS DS가 효과적으로 감소함을 확인하였습니다.
* 단순 SNR 최대화(Benchmark) 방식보다, 제안된 CP 오버헤드 반영 방식이 다양한 환경에서 더 높은 유효 데이터 전송률을 기록하였습니다.

### 3.2. DQN 에이전트 학습 및 성능 평가 (Python)
* **학습 수렴:** 1,000 에피소드 중 약 300 에피소드 이후부터 ISI 발생을 회피하고 최적 CP 구간을 유지하는 정책을 학습하여 보상이 양수로 안정화되었습니다.
* **성능 향상:** 테스트 스텝에서 에이전트는 보수적인 기준치(DS의 3배)보다 짧은 CP를 설정하여 오버헤드를 적극적으로 줄였습니다. 그 결과, 이론적 기준 대비 평균 약 1.8배 높은 Spectral Efficiency (SE)와 향상된 전력 효율을 달성하였습니다.

<img width="1595" height="936" alt="image" src="https://github.com/user-attachments/assets/f118e2f1-5c1f-4e4a-8149-63148690bacf" />
*(그림 4. DQN 기반 적응형 CP 알고리즘 성능 대시보드_에피소드별 학습 보상 및 테스트 스텝별 SE, 전력 효율 비교_v1 기준)*

<br><br>
<img width="2984" height="2165" alt="adaptive_cp_dashboard" src="https://github.com/user-attachments/assets/529f07d4-7877-48ec-9256-733da1c8fd4e" />
*(그림 5. DQN 기반 적응형 CP 알고리즘 성능 대시보드_v4 기준)*

## 4. 설치 및 실행 가이드 (Getting Started)

### Prerequisites
* Python 3.11
* PyTorch, NumPy, SciPy, Matplotlib

### Repository Structure
env.py: 강화학습 채널 환경 생성 및 상태 전이 물리 연산

DQN.py: PyTorch 기반 DQN 신경망 모델 구조

train.py: 에이전트 학습 진행 및 가중치 저장

test.py: 학습된 모델을 활용한 평가 및 성능 대시보드 시각화

### 환경 변수 설정
python env.py

### DQN 모델 구축
python DQN.py

### 모델 학습 실행
python train.py

### 학습된 모델 성능 평가 및 시각화
python test.py
