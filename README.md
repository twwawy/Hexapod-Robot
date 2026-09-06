<div align="center">

# 모듈형 재난대응 6족 로봇

### Physical AI 기반 험지 이동·환경 인식·임무 확장 플랫폼

[![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano%20Super-76B900?style=flat-square&logo=nvidia&logoColor=white)](SW/Jetson/README.md)
[![MCU](https://img.shields.io/badge/MCU-STM32F446RE-03234B?style=flat-square&logo=stmicroelectronics&logoColor=white)](SW/STM32/STM32F446RE%20설정%20정리본.md)
![DOF](https://img.shields.io/badge/Mobility-6%20Legs%20%7C%2018%20DOF-00B8A9?style=flat-square)
[![Control](https://img.shields.io/badge/Control-MATLAB%20%2F%20Simulink-F37021?style=flat-square&logo=mathworks&logoColor=white)](SW/Controller/Controller_Architecture.md)
![Simulation](https://img.shields.io/badge/Simulation-Isaac%20Sim%20%7C%20MuJoCo-7A5AF8?style=flat-square)

**험지개척단 · 인하대학교 · 2026 종합설계경진대회**

<img src="docs/assets/hexapod-prototype.jpg" alt="모듈형 재난대응 6족 로봇 시제품" width="880">

</div>

## 통합 v4: 시작과 문서

현재 작업 브랜치 `codex/adaptive-hybrid-rl-integration`은 최신 main에서 시작해 adaptive MJX를
선택적으로 가져왔다. Full merge는 하지 않았다. **코드 통합과 portable 계약 검사까지 진행했으며,
계단 등판·PPO·STM32 보드 동작은 아직 검증하지 않았다.**

| 카테고리 | 문서 / 실행 경로 |
|---|---|
| 통합 architecture·24-D/관측·변경 내용·남은 위험 | [v4 통합 설명](docs/ADAPTIVE_INTEGRATION_V4.md) |
| 브랜치 분석·파일 출처 | [사전 분석](docs/ADAPTIVE_INTEGRATION_ANALYSIS.md), [전체 변경 파일](docs/ADAPTIVE_INTEGRATION_FILES.md) |
| Stage0 viewer·표시·조작 | [viewer 사용법](docs/HEXAPOD_MJX_ADAPTIVE_GAIT_USAGE.md) |
| 학습 단계 | [학습 계획](docs/HEXAPOD_MJX_ADAPTIVE_GAIT_LEARNING_PLAN.md) |
| 짧은 평지 cycle 반복·cycle별 W&B score/영상 | [실행 명령 및 저장 규칙](docs/ADAPTIVE_OBSERVE_CYCLES.md) |
| Rough hfield 오류 수정·평지 가중치에서 재개 | [adaptive box 지형과 명시적 이전](docs/ADAPTIVE_ROUGH_BOXES.md) |
| 수 회 Tripod 후 정지·영상 종료 수정 | [원인·수정·사용자 검증 명령](docs/TRIPOD_STOP_FIX.md) |
| Jetson ↔ STM32 wire | [SPI v3](docs/ADAPTIVE_SPI_V3.md) |
| STM32 실제 contact/Wave 기준 | [Wave gait](SW/STM32/WAVE_GAIT.md) |
| 기존 18-D replay | `scripts/view_trained_policy.sh`, 기존 `mjx/firmware_mjx_controller.py` |

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/view_foothold_planner.sh --controller adaptive --terrain flat --perception oracle
bash scripts/view_foothold_planner.sh --controller adaptive --terrain steps --perception oracle
bash scripts/view_foothold_planner.sh --controller adaptive --terrain steps --perception lidar --stage0 --speed 0.04
```

먼저 flat의 zero-action Tripod, 계단의 candidate/reference/selected/latched marker,
short-step·Wave 전환과 unknown HOLD를 확인한다. 이후 학습/오프라인 Jetson/ARM 빌드는
[순서별 명령](docs/ADAPTIVE_INTEGRATION_V4.md#사용자-실행-순서)을 사용한다.
방향키 포커스 문제가 있으면 `W/S`로 전후, `A/D`로 회전할 수 있다. 키가 들어오면 콘솔에
`command: vx=...`가 즉시 출력된다. `--speed 0.04`는 키 입력 없이 시작 명령을 넣는다.

## 프로젝트 소개

본 프로젝트는 붕괴 구조물, 잔해, 경사면, 단차와 계단처럼 사람이 직접 접근하기 어려운 재난 현장을 선행 탐색하기 위한 6족 보행 로봇이다. 하나의 공통 보행 플랫폼 위에 센서와 작업 장비를 교체할 수 있는 모듈형 구조를 적용하여, 환경 매핑·자율 탐색·현장 조작·구호물품 전달 등 임무별 구성을 목표로 한다.

6개의 독립적인 3자유도 다리와 총 18개의 관절을 사용하며, **Geometry가 지형과 가능한 영역을 계산하고, 강화학습이 foothold·clearance·몸체 자세/높이·보폭·trajectory timing을 최적화하며 STM32가 접촉과 실행 안전을 담당하는 계층형 제어 구조**를 채택하였다. Jetson은 SLAM·환경 인식·경로계획·강화학습 등 상위 연산을, STM32는 200 Hz 주기의 발끝 궤적·역기구학·서보 제어를 담당하도록 분리하였다.

> [!NOTE]
> 이 저장소는 구현 코드, 설계 자료, 시뮬레이션 결과와 개발 문서를 함께 관리한다. 기능별 완성도와 검증 수준이 다르므로 아래의 **현재 구현 상태**와 각 문서의 검증 범위를 확인해야 한다.

## 핵심 특징

| 영역 | 주요 내용 |
|---|---|
| **험지 보행** | Tripod Gait, 3차 Bézier 발끝 궤적, 3자유도 역기구학, 18개 서보모터 제어 |
| **접촉 적응** | 6개 FSR 접촉 상태 기반 Early Landing·Late Landing 보정과 작업영역 제한 |
| **계층형 Physical AI** | Classical Controller의 발끝 목표에 제한된 Cartesian Residual을 더하는 강화학습 구조 |
| **환경 인식** | Livox MID-360, RealSense D435, IMU를 이용한 3차원 인식·SLAM·경로계획 구조 |
| **분산 제어** | Jetson 상위 판단과 STM32 실시간 보행 제어 분리, 128바이트 SPI v3 실행/관측 프레임 |
| **모듈형 임무 확장** | 센서 위치 변경, 열화상·가스 센서, 매니퓰레이터, 구급·적재 모듈 확장 고려 |
| **원격 운용** | 위치·자세·통신·이동 경로 모니터링과 목적지·모드·비상 정지 명령 구조 |

## 시스템 구성

<div align="center">
  <img src="docs/assets/system-architecture.jpg" alt="Jetson과 STM32 기반 분산 제어 시스템 구조" width="920">
</div>

```text
LiDAR / Depth Camera / IMU / 운용자 명령
                    │
                    ▼
Jetson Orin Nano Super
SLAM → 환경 인식 → 경로계획 → 강화학습·상위 의사결정
                    │  목표 속도·발 목표·임무 명령
                    ▼
STM32 NUCLEO-F446RE
고정 제어 주기 → 발끝 궤적 → 3DOF IK → 18개 서보 제어
                    │
                    ▼
              6족 로봇 구동
                    │  관절각·접촉·IMU 상태
                    └──────────────────────► Jetson / 관제 시스템
```

### STM32 ↔ Jetson 통신

통합 브랜치는 **SPI v3, 128 bytes**를 기본으로 사용한다. 최종 AdaptiveExecutionPlan을
explicit little-endian fixed point로 전달하고 CRC·session·sequence·observation age·plan을 검증한다.
상태와 계획 상세는 같은 시점의 두 observation page로 제공한다. 기존 v2 codec은 보존하지만
v2 master와 기본 v3 app은 호환되지 않는다. 실제 DMA/DRDY 운용은 사용자 검증 대상이다.

- [SPI v3 정확한 offset·단위·CRC](docs/ADAPTIVE_SPI_V3.md)
- [기존 v2 문서](SW/STM32/STM32-Jetson%20SPI%2032바이트%20패킷%20프로토콜.md)

## 보행 및 지형 적응 제어

### 1. Classical Controller

세 다리가 지면을 지지하는 동안 나머지 세 다리를 이동시키는 Tripod Gait를 사용한다. 스윙 발의 경로는 3차 Bézier 곡선으로 생성하고, 발끝 목표 좌표를 각 다리의 3자유도 역기구학으로 관절 목표각으로 변환한다. 작업영역과 IK 유효성을 검사하여 비정상적인 관절 명령이 출력되지 않도록 제한한다.

### 2. FSR 접촉 보정

- **Early Landing:** 예상보다 먼저 접촉하면 스윙을 종료하고 지지 상태로 전환
- **Late Landing:** 예정된 착지 시점에도 미접촉이면 발끝을 추가 하강하여 지면 탐색
- **기준 높이 보정:** 접촉 보정이 반복되어 몸체 높이가 누적 변화하지 않도록 기준값 보정

### 3. Adaptive residual policy 24-D v4

Wide25 후보에서 safe reference를 찾고, 별도 local25 후보에서 RL의 **X ±6 cm / Y ±4 cm**
요청을 안전한 endpoint로 projection한다. Terrain Z는 geometry가 결정한다.
6개 clearance, roll/pitch/body height, stride 및 apex/transfer timing도 policy가 조절한다.
Supervisor는 **Tripod normal → short-step → Wave → HOLD** 순서로 결정한다.
UNKNOWN은 즉시 Wave로 보내지 않는다. 학습 가중치 없이 zero-action Stage0부터 확인한다.

## 모듈형 재난 대응 시나리오

하나의 6족 플랫폼을 임무에 맞춰 재구성하는 것이 프로젝트의 핵심 방향이다.

| 구성 | 역할 | 현재 범위 |
|---|---|---|
| **LiDAR·IMU 모듈** | 3D 지도 생성, 위치·자세 추정, 자율 탐색 | 하드웨어·시스템 구조 개발 |
| **센서 위치 변경** | 고정형 LiDAR의 사각지대 보완, 목적별 공간 정보 수집 | 장착 구조 확장 방향 |
| **매니퓰레이터** | 문 개방, 가벼운 장애물 이동, 현장 작업 | 확장 모듈·실물 통합 검증 필요 |
| **구급·적재 모듈** | 구급상자, 센서와 필수 물품 전달 | 확장 제안 |
| **열화상·가스 센서** | 인명·열원 탐색, 유해 환경 측정 | 확장 제안 |

```text
현장 투입 → 환경 인식·지도 생성 → 통과 가능성 판단
         → 자율주행·험지 이동 → 현장 조작 또는 물자 전달
         → 상태 보고·필요 시 원격 개입
```

## 현재 구현 및 검증 상태

| 항목 | 상태 | 검증 범위 |
|---|---|---|
| 6족 기구·18자유도 구동부 | 제작 | Fusion 360, 분리형 다리·몸체·센서 장착부 |
| STM32 기본 보행 제어 | 구현 | Tripod Gait, Bézier 궤적, IK, Safety, FSR 접촉 처리 |
| STM32 실시간 로깅 | 구현 | 접촉 상태, 보행 Phase, 발끝 목표, IK 결과, 관절 목표각 |
| STM32–Jetson 상태 통신 | STM32 측 구현 | SPI v3 codec/app 연결, 실제 DMA/DRDY는 미검증 |
| MATLAB/Simulink 제어 시험 | 실행·기록 | 몸체 고정 모델의 0–81 s 단일 시험, 실제 자유 보행 성능 아님 |
| Isaac Sim·MuJoCo | 시뮬레이션 구성 | 보행·Residual-RL 학습 구조, 실물 성능과 별도 검증 필요 |
| Jetson SLAM·자율주행 | 설계·통합 대상 | geometry bridge·zero-action·SPI codec 구현, live 센서/odom 연결은 별도 |
| 원격 관제·LoRa | 시스템 설계 | 실시간 현장 운용 성능은 별도 검증 필요 |

### 공개된 Simulink 시험 결과

2026-08-31에 기존 `plant.slx`를 MATLAB R2026a Update 3에서 실행한 결과다. 몸체가 월드에 고정된 관절·기구학 제어 시험이므로 실제 험지 주행 성능으로 해석하면 안 된다.

| 평가 항목 | 결과 |
|---|---:|
| 6개 다리 통합 발끝 RMS 거리 오차 | **3.385 mm** |
| 발끝 최대 거리 오차 | **8.631 mm** |
| 18개 관절 통합 RMS 오차 | **0.4004°** |
| 6개 다리 IK 유효 표본 비율 | **100%** |

<div align="center">
  <img src="SW/Controller/Simulink/evaluation/figure_07_foot_trajectory.png" alt="MATLAB Simulink 발끝 궤적 및 추종 결과" width="880">
</div>

자세한 조건, 원시 데이터와 재현 절차는 [MATLAB/Simulink 실행 결과](SW/Controller/Simulink/evaluation/RESULTS.md)를 참고한다. 평지 10 m 보행 성공률, 실제 평균 보행 속도, 최대 단차·경사, 계단 통과율, 실제 경로 추종 오차와 Residual-RL 전후 비교는 현재 공개 자료만으로 확정하지 않았으며 임의 수치를 제시하지 않는다.

## 저장소 구성

```text
Hexapod-Robot/
├─ HW/
│  ├─ PCB/                  # PCB Gerber·Drill 자료
│  ├─ stl_file/             # 3D 프린팅용 STL
│  ├─ gcode_file/           # G-code·3MF
│  └─ urdf/                 # 현재 URDF·시뮬레이션 설정
├─ SW/
│  ├─ Controller/           # 보행 제어 수식·좌표계·Simulink
│  ├─ STM32/                # STM32CubeIDE 펌웨어·통신 문서
│  └─ Jetson/               # 상위 제어 범위·인터페이스 문서
└─ docs/
   ├─ assets/               # README 이미지
   └─ 2026-inha-capstone-final-report-public.pdf
```

## 주요 문서

- [2026 인하 종합설계경진대회 최종보고서 - 공개용](docs/2026-inha-capstone-final-report-public.pdf)
- [하드웨어 구성 및 제작 자료](HW/README.md)
- [하드웨어 부품 목록](HW/parts.md)
- [제어기 Architecture](SW/Controller/Controller_Architecture.md)
- [제어기 상세 설계](SW/Controller/Controller_detail.md)
- [좌표계와 관절 정의](SW/Controller/좌표축/README.md)
- [STM32F446RE 설정](SW/STM32/STM32F446RE%20설정%20정리본.md)
- [STM32 코드 구조](SW/STM32/STM32%20코드%20구조%20정리본.md)
- [STM32–Jetson SPI 프로토콜](SW/STM32/STM32-Jetson%20SPI%2032바이트%20패킷%20프로토콜.md)
- [매니퓰레이터 UART 프로토콜](SW/STM32/Manipulator_UART_Protocol.md)
- [Jetson 소프트웨어 범위](SW/Jetson/README.md)
- [Simulink 평가 결과 및 재현 절차](SW/Controller/Simulink/evaluation/RESULTS.md)

## 주의 사항

- 고토크 서보모터와 LiPo 배터리를 사용하는 시스템이므로 전원·배선·비상 정지 절차를 먼저 확인해야 한다.
- 소스 코드, 빌드 결과물, 실제 보드에 플래시된 펌웨어와 실물 시험 결과는 서로 다른 검증 단계다.
- 보고서와 시뮬레이션의 미평가 항목은 실패나 0%를 뜻하지 않으며, 반복 실물 시험 후 수치화해야 한다.

---

<div align="center">

**One hexapod platform, multiple disaster-response missions.**

</div>
