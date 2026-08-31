# MATLAB/Simulink 실행 결과

2026-08-31에 기존 `plant.slx`를 MATLAB R2026a Update 3에서 직접 실행했다. 원래 입력과 제어 상수를 유지한 **0–81초 시험 1회**이며, 실제 실행에는 약 105.6초가 걸렸다. 202개 신호를 기록했다. 모델과 파라미터 원본은 수정하지 않았다.

**이번 결과는 몸체가 월드에 고정된 관절·기구학 제어 시험이다. 실제 자유 보행이나 험지 주행 성능을 입증하는 결과는 아니다.**

**최신 요청 반영:** 기존 실행의 VirtualIMU 값으로 가상 자세 RMSE 표와 사진 8을 채웠다. 아래 그림과 표에 가상 신호임을 명시하며, 실제 몸체의 보행 안정성은 여전히 미평가이다. [가상 자세 비교와 해석 범위](BODY_STABILITY_COMPARISON.md)에 계산 구간과 문헌 기준값을 정리했다.

## 보고서에 넣을 그림

### 사진 7. MATLAB/Simulink 발끝 궤적 및 추종 결과

[PNG 원본](figure_07_foot_trajectory.png) · [MATLAB 편집 파일](figure_07_foot_trajectory.fig)

권장 캡션:

> 사진 7. MATLAB/Simulink 기반 6족 로봇의 발끝 궤적 및 추종 결과. 기존 모델의 전진 명령 구간(7 ≤ t < 10 s)에서 발끝 목표와 관절 동역학 응답으로 계산한 정기구학 위치를 비교하였다. 6개 다리의 3차원 발끝 RMS 오차는 3.39 mm이다. 몸체 고정 모델에서 얻은 제어 시험 결과이며 지형 통과 성능을 의미하지 않는다.

왼쪽 위는 1번 다리의 8–9초 궤적이며, 나머지 패널은 전진 명령 구간 전체를 사용한다. 좌표는 몸체 좌표계이다. 목표는 `BodyPosturePIOverlay` 출력, 응답은 `FK_Leg2Body` 출력이다. 접촉 센서로 측정한 발 위치가 아니다.

### 기존 제어기 결과 보조 그림

[PNG 원본](classical_control_results.png) · [MATLAB 편집 파일](classical_control_results.fig)

권장 캡션:

> 기존 Classical Controller의 가상 자세 응답 및 관절 추종 결과. 상단은 81초 시험의 Roll·Pitch 기준과 VirtualIMU 출력을, 하단은 전진 구간의 대표 관절 응답과 18개 관절의 RMS 오차를 나타낸다. VirtualIMU는 자세 명령에서 생성되므로 실제 몸체 흔들림이나 자세 안정성을 검증한 결과로 해석할 수 없다.

### 사진 8. Classical Controller 가상 자세와 기존 6족 로봇의 참고값

[PNG 원본](figure_08_body_attitude_comparison.png) · [MATLAB 편집 파일](figure_08_body_attitude_comparison.fig) · [수치 CSV](body_attitude_comparison.csv) · [가상 IMU 요약](virtual_imu_metrics.json)

| 대상 | Roll RMSE (°) | Pitch RMSE (°) | 데이터 유형 |
|---|---:|---:|---|
| 우리 Classical | 0.000 | 0.000 | 명령 기반 VirtualIMU, 7 ≤ t < 10 s |
| HexWalker II, Tripod | 0.390 | 0.366 | 문헌의 실제 로봇 시험 |

우리 가상 출력은 이 전진 구간에서 0°로 유지되므로 RMSE가 0이다. 실제 몸체 안정성이나 외부 로봇 대비 우위를 뜻하지 않는다.

> 사진 8. 기존 Classical Controller의 명령 기반 VirtualIMU에서 산출한 Roll·Pitch RMSE와 HexWalker II의 문헌 실측 참고값. 우리 값은 기존 Simulink 실행의 전진 명령 구간(7 ≤ t < 10 s), 문헌값은 Zhang et al. (2021), Table 5의 Tripod 실험을 사용했다. 가상 신호와 실제 몸체 측정값은 데이터 생성 방식과 시험 조건이 다르므로 실제 보행 안정성의 우열이나 개선율을 나타내지 않는다.

재생성 명령은 `make_body_stability_figure`이다. 기존 원본 MAT를 후처리하며 새 보행 시험이나 동역학 모델 실행을 수행하지 않는다.

### 이전 사진 8 — 속도·추종 지표 참고 그림

[PNG 원본](figure_08_hexapod_comparison.png) · [MATLAB 편집 파일](figure_08_hexapod_comparison.fig) · [비교 수치 CSV](hexapod_comparison.csv) · [출처와 비교 조건](COMPARISON_SOURCES.md)

사용자의 변경 요청에 따라 **현재 Classical Controller의 시뮬레이션 측정값과 대표 6족 로봇의 공개 문헌 수치**를 비교한다. Residual-RL 비교나 설계 구조도가 아니다.

권장 캡션:

> 사진 8. 현재 Classical Controller의 제어 정확도와 대표 6족 로봇의 문헌 성능 비교. 현재 모델의 발끝·관절 RMS 추종 오차는 각각 3.385 mm와 0.400°이며, RHex·PhantomX AX·HAntR의 보고된 최대 주행 속도는 각각 0.55·0.29·0.43 m/s이다. 모델과 시험 조건이 다르므로 직접적인 우열 판단에는 사용할 수 없다.

RHex 속도는 [Saranli et al. (2001), Table 1](https://www.ri.cmu.edu/pub_files/pub4/saranli_uluc_2001_1/saranli_uluc_2001_1.pdf), PhantomX AX 및 HAntR 속도는 [Čížek et al. (2021), Table 1](https://comrob.fel.cvut.cz/papers/access21hantr.pdf)에 근거한다. 해당 논문의 로봇 세대와 시험에서 보고한 값으로, 각 플랫폼의 최신 최고 성능을 뜻하지 않는다.

현재 모델의 0.2349 m/s는 위치 추정기 출력에서 계산한 평균값이므로 **실제 주행 최대 속도 막대에서는 제외**했다. 문헌 로봇의 동일 정의 RMS 수치를 확보하지 못한 곳은 ‘미확보’, 현재 모델의 자유 주행 시험은 ‘미평가’로 표시한다. 결측값은 0이나 성능 부족을 의미하지 않는다.

## 실제로 산출한 제어 시험 수치

| 평가항목 | 결과 | 산출 범위 |
|---|---:|---|
| 6개 다리 통합 발끝 RMS 거리 오차 | **3.385 mm** | 전진 명령 구간, 7 ≤ t < 10 s |
| 발끝 최대 거리 오차 | **8.631 mm** | 같은 구간, 6개 다리 중 최대 |
| 18개 관절 통합 RMS 오차 | **0.4004°** | 같은 구간, 속도 제한 후 목표 대비 |
| 6개 다리 모두 IK 유효인 표본 비율 | **100%** | 0–81 s, 200 Hz 평가 표본 |
| 제어기 Fault 발생 | **없음** | 이번 시험 1회 |
| 가상 전복 Fault 발생 | **없음** | 실제 전복 시험이 아님 |

발끝 오차는 각 다리의 `norm(응답 XYZ - 목표 XYZ)`이다. 통합 RMS는 여섯 다리와 평가 시간 표본 전체의 제곱평균제곱근으로 계산했다. 관절 통합 RMS는 18개 관절의 라디안 오차를 도 단위로 변환해 같은 방식으로 계산했다. 목표 신호에 시간 지연 보정이나 위상 맞춤을 적용하지 않았다.

| 다리 | 발끝 RMS 오차 (mm) | 최대 오차 (mm) | 발끝 Z 변화폭 (cm) |
|---|---:|---:|---:|
| 1 | 3.256 | 8.283 | 21.918 |
| 2 | 3.331 | 8.319 | 21.958 |
| 3 | 3.542 | 8.631 | 22.033 |
| 4 | 3.298 | 8.271 | 21.919 |
| 5 | 3.308 | 8.326 | 21.954 |
| 6 | 3.562 | 8.622 | 22.007 |

발끝 Z 변화폭은 몸체 좌표계에서 전진 구간의 `max(Z)-min(Z)`이다. **약 22 cm라는 값을 최대 극복 단차로 사용하면 안 된다.**

## 요청한 최종 성능표의 현재 평가 상태

| 평가항목 | 최종 결과 | 이유 |
|---|---|---|
| 평지 10 m 보행 성공률 | 미평가 | 자유 보행 10 m 시험 및 10회 반복 없음 |
| 평균 보행 속도 | 미평가 | 실제 몸체 이동량을 측정할 수 없는 고정 몸체 모델 |
| 최대 극복 단차 | 미평가 | 단차와 접촉 동역학을 이용한 통과 시험 없음 |
| 최대 등판 경사 | 미평가 | 경사면 주행 시험 없음 |
| 계단 통과 성공률 | 미평가 | 계단 시험 및 10회 반복 없음 |
| 최대 Roll 변동 **(가상 신호)** | **33.677°** | 전체 0–81 s 최대–최소, 의도적 자세 변경 포함; 실제 몸체 미평가 |
| 최대 Pitch 변동 **(가상 신호)** | **34.965°** | 전체 0–81 s 최대–최소, 의도적 자세 변경 포함; 실제 몸체 미평가 |
| Classical Controller 험지 성공률 | 미평가 | 지형·성공 판정·반복 시험 없음 |
| Residual-RL 험지 성공률 | 미평가 | 학습 정책과 비교 시험을 확인하지 못함 |
| 평균 경로 추종 오차 | 실제 경로 기준 미평가 | 내부 위치 추정값은 있으나 독립적인 실제 몸체 경로 없음 |
| 장애물 탐지 성공률 | 미평가 | 이 모델에 탐지 평가 입력·정답 데이터 없음 |
| 장애물 회피 성공률 | 미평가 | 이 모델에 회피 경로·충돌 판정 시험 없음 |

미평가는 0%나 실패를 뜻하지 않는다. 이 항목들을 임의 숫자로 채우지 않았다.

## 가상·추정 신호 참고값

아래 값은 CSV에서 실제로 계산됐지만 위 표의 실보행 성능을 대체하지 못한다.

| 내부 신호 평가항목 | 값 | 의미 |
|---|---:|---|
| 위치 추정기 기준 전진 평균 x 속도 | 0.2349 m/s | 7.000–9.995 s의 추정 x 변위 / 경과 시간 |
| 내부 XY 위치 기준–추정값 평균 차이 | 0.01004 m | 7 ≤ t < 10 s, 두 축 거리 평균 |
| 가상 Roll 최대 절댓값 | 18.0789° | 0–81 s, 의도적인 자세 변경 포함 |
| 가상 Pitch 최대 절댓값 | 27.8354° | 0–81 s, 의도적인 자세 변경 포함 |
| 가상 Roll 최대–최소 변화폭 | 33.6774° | 0–81 s, peak-to-peak |
| 가상 Pitch 최대–최소 변화폭 | 34.9651° | 0–81 s, peak-to-peak |

`Plant/VirtualIMU`는 지연된 자세 명령을 Roll·Pitch 출력으로 반환한다. `Plant/MATLAB Function1`의 `TestContact`는 보행 위상과 시간으로 접촉 신호를 생성한다. `World Frame → Rigid Transform4 → Rigid Transform7 → Body` 연결에는 몸체가 자유 이동할 관절이 없다. 따라서 위치 추정기가 이동량을 출력하더라도 실제 몸체 이동량은 아니다. 이 모델의 수치는 별도로 변경된 STM32 펌웨어 상수와도 구분해야 한다.

## 데이터와 재현

| 파일 | 내용 |
|---|---|
| `simulation_raw.mat` | Simulink SimulationOutput 원본, 신호 목록, 실행 소요 시간 |
| `signals_200Hz.csv` | 0–81초, 0.005초 간격의 202개 신호와 시간 열 |
| `signal_map.csv` | 각 CSV 열의 원본 블록 경로와 출력 포트 |
| `foot_metrics.csv` | 다리별 발끝 RMS·최대 오차·Z 변화폭 |
| `joint_metrics.csv` | 18개 관절의 RMS 오차, 다리 1의 관절 1–3부터 순서대로 배열 |
| `metrics.json` | 조건과 한계를 포함한 주요 수치 |
| `simulation_log.txt` | 실행 시작·종료 및 MATLAB 버전 기록 |
| `model_functions.txt` | 실행 당시 SLX 내부 MATLAB 함수의 읽기 전용 추출본 |
| `run_evaluation.m` | 기존 모델을 저장하지 않고 로깅을 추가해 실행 |
| `export_evaluation.m` | 저장된 MAT에서 CSV·수치·그림 재생성 |

CSV와 위 수치는 원시 신호를 200 Hz 공통 시간축으로 선형 보간해 계산했다. 중복 시각은 마지막 값을 사용했고, 전체 신호의 시간 범위와 유한수 여부를 검사했다. 원래 가변 시간 간격의 데이터는 MAT에 보존했다. 후처리에는 추가 Statistics Toolbox가 필요하지 않다.

MATLAB에서 이 폴더를 현재 폴더로 선택한 뒤 실행한다.

```matlab
run_evaluation
export_evaluation
```

그림만 다시 만들 때는 `export_evaluation`만 실행한다. `plant.slx`가 이미 열려 있다면 미저장 작업을 먼저 보관하고 별도의 MATLAB 세션에서 실행한다. 실행 스크립트는 모델을 저장하지 않고 닫는다.

실행에 사용한 원본의 SHA-256:

- `plant.slx`: `10B2DF43606460FBF9658EE338C4FB42BB2C1FBD3513F292A4D7DAAB6C56ABF3`
- `plant_parameters.m`: `46E7ABFF286E2CE896B79EB7C3372D5CD4C5525D4DFC44FF4FF3D457DB767943`
