# 6족 로봇 보행 제어기 상세 설계

[Controller_Architecture.md](Controller_Architecture.md)의 제어 구조를 수식과 이산시간 규칙으로 정의한다. 조종기 채널·시험 입력은 [드론 조종기 입력 README](드론%20조종기%20입력/README.md), 좌표 부호와 기본 자세는 [좌표축 README](좌표축/README.md)를 따른다.

이 문서는 현재 제어 모델의 동작 기준이다. 블록 배치나 선 연결 방법은 다루지 않는다.

## 1. 공통 정의

### 1.1 좌표계

| 기호 | 의미 |
|---|---|
| \(\{W\}\) | 지면에 고정된 절대 좌표계 |
| \(\{B\}\) | 현재 몸체 원점 좌표계 |
| \(\{R\}\) | 명목 몸체 기준 좌표계 |
| \(\{L_i\}\) | i번 다리 로컬 좌표계 |
| \(O_B\) | 현재 몸체 원점 |
| \(O_R\) | 기본 자세의 몸체 기준점 |
| \(F_i\) | i번 다리 발끝 |

몸체 축은 +X 전진, +Y 왼쪽, +Z 위쪽이다.

몸체 선속도, 각속도와 Body Twist는 다음과 같다.

\[
v_B=
\begin{bmatrix}
v_x\\v_y\\v_z
\end{bmatrix},
\qquad
\omega_B=
\begin{bmatrix}
\omega_x\\\omega_y\\\omega_z
\end{bmatrix},
\qquad
\xi_B=
\begin{bmatrix}
v_B\\\omega_B
\end{bmatrix}
\]

벡터 \(a\)의 Skew-Symmetric Matrix는 다음과 같다.

\[
[a]_\times=
\begin{bmatrix}
0&-a_z&a_y\\
a_z&0&-a_x\\
-a_y&a_x&0
\end{bmatrix},
\qquad
a\times b=[a]_\times b
\]

### 1.2 제어 주기

전체 제어 루프는 200 Hz로 실행한다.

\[
T_s=0.005\ \mathrm{s}
\]

센서 Snapshot부터 서보 명령 갱신까지 한 주기 안에서 순서대로 완료한다고 가정한다. 계산이 일찍 끝나면 다음 5 ms Tick까지 기다린다. 별도의 연산시간 초과 Fault는 현재 구현하지 않는다.

## 2. 사용자 입력 전처리

### 2.1 정규화와 Dead Zone

Throttle·Yaw·Roll·Pitch의 공통 raw 범위는 -1000~1000이고 중립은 0이다. 정규화 값은 \(u\in[-1,1]\)이다.

Dead Zone \(\delta\)는 다음처럼 적용한다.

\[
D(u)=
\begin{cases}
0, & |u|\le\delta\\
\operatorname{sgn}(u)\dfrac{|u|-\delta}{1-\delta}, & |u|>\delta
\end{cases}
\]

현재 raw 기준 Dead Zone은 다음과 같다.

| 입력 | Dead Zone |
|---|---:|
| Throttle | ±20 |
| Yaw·Roll·Pitch | ±50 |

Dead Zone 밖의 범위는 다시 -1~1로 선형 대응한다.

### 2.2 Low Pass Filter

Dead Zone 이후 각 축에 5 Hz 1차 LPF를 적용한다.

\[
u_f[k]=\alpha u_f[k-1]+(1-\alpha)u_d[k]
\]

\[
\alpha=e^{-2\pi f_cT_s},
\qquad
f_c=5\ \mathrm{Hz}
\]

모드가 바뀌면 새 모드 출력 계산 전에 네 필터 상태를 모두 0으로 초기화한다. 조종 모드에서 SA가 바뀌면 Yaw 필터만 0으로 초기화한다.

### 2.3 연속 명령

임시 조종기 입력과 강화학습 Action의 raw 변화율은 5000/s로 제한한다. 5 ms당 최대 변화량은 25 raw이며 중립에서 최대까지 0.2 s, 최소에서 최대까지 0.4 s가 걸린다.

실제 RC와 강화학습 모두 속도·자세 명령 경로를 사용한다. Step Action을 발끝 위치나 관절각에 직접 전달하지 않는다.

## 3. 상위 상태

### 3.1 상태 우선순위

\[
KILL
>
ROLLOVER\_FAULT
>
CONTROLLER\_FAULT
>
LANDING
>
STANDING
>
READY
\]

READY 안에서 MANUAL 또는 CORRECTION을 선택한다. ARM과 AUTONOMOUS는 조종기 배치만 유지하며 현재 제어 계산에는 포함하지 않는다.

### 3.2 서기와 착지

서기는 자세 진행률을 5.6 s 동안 0에서 1로 증가시킨 뒤 0.5 s 동안 안정 상태를 유지하면 완료한다.

착지는 다음 순서를 사용한다.

1. 보행 중이었다면 모든 Swing 발의 접촉을 완료한다.
2. Leg 1·3·5와 Leg 2·4·6을 각각 0.5 s 동안 순서대로 정리한다.
3. 5.6 s 동안 자세 진행률을 1에서 0으로 낮춘다.
4. 0.5 s 동안 안정 상태를 유지한 뒤 LANDED로 전환한다.

보행하지 않았다면 발 정리 단계를 건너뛰고 자세 하강부터 시작한다.

### 3.3 동작 허가

READY 진입 직후에는 동작을 허가하지 않는다. 네 짐벌이 다음 범위에서 0.2 s 동안 중립이어야 Motion Armed를 활성화한다.

| 입력 | 중립 범위 |
|---|---:|
| Throttle | ±20 raw |
| Yaw·Roll·Pitch | ±50 raw |

READY가 아니면 Motion Armed를 즉시 해제한다. CRSF 100 ms Timeout과 재연결 조건은 드론 조종기 입력 문서를 따른다.

## 4. 사용자 Body Command

필터 출력 \(u_{T,f},u_{Y,f},u_{R,f},u_{P,f}\)를 사용한다.

### 4.1 조종 모드

\[
v_{x,user}=0.28u_{T,f}
\]

\[
\phi_{ref}=45^\circ u_{R,f},
\qquad
\theta_{ref}=45^\circ u_{P,f}
\]

SA OFF에서는 다음을 사용한다.

\[
v_{y,user}=0,
\qquad
\omega_{z,user}=45^\circ/\mathrm{s}\;u_{Y,f}
\]

SA ON에서는 다음을 사용한다.

\[
v_{y,user}=0.28u_{Y,f},
\qquad
\omega_{z,user}=0
\]

단위는 선속도 m/s, 각도 deg와 각속도 deg/s이다.

### 4.2 Heading Reference

SA OFF에서 목표 Heading은 사용자 Yaw 각속도를 적분한다.

\[
\psi_{ref}[k+1]
=
\operatorname{wrap}_{\pi}
\left(
\psi_{ref}[k]+\omega_{z,user}[k]T_s
\right)
\]

Yaw 입력이 중립이면 마지막 Heading을 유지한다. MANUAL 진입 또는 SA 전환 시에는 현재 IMU Yaw를 새 목표로 사용한다.

\[
\psi_{ref}\leftarrow\psi_{IMU}
\]

SA ON에서는 사용자 Yaw 각속도를 0으로 두고 Heading PI의 보정만 허용한다.

### 4.3 보정 모드

\[
v_{x,corr}=0.05u_{P,f},
\qquad
v_{y,corr}=0.05u_{R,f},
\qquad
v_{z,corr}=0.05u_{T,f}
\]

\[
\psi_{corr,ref}=10^\circ u_{Y,f}
\]

보정 Yaw는 각속도가 아니라 상대 목표각이다. Yaw 입력이 중립으로 돌아오면 목표각도 0 deg로 복귀한다.

## 5. 몸체 위치 추정

i번 다리의 측정 관절각을 \(q_i\)라 하면 몸체 좌표계 발끝 위치는 FK로 계산한다.

\[
{}^Bp_{F_i}^{meas}=FK_i(q_i)
\]

STANCE이면서 CONTACT가 확인된 다리만 원시 유효 다리로 사용한다. Stance 시작 순간의 지면 고정 Anchor는 다음과 같다.

\[
{}^Wp_{F_i}^{anchor}
=
{}^W\hat p_B
+
{}^WR_B
{}^Bp_{F_i}^{meas}
\]

Stance 중에는 Anchor를 갱신하지 않는다. i번 다리로 계산한 몸체 위치 후보는 다음과 같다.

\[
{}^W\hat p_{B,i}
=
{}^Wp_{F_i}^{anchor}
-
{}^WR_B
{}^Bp_{F_i}^{meas}
\]

### 5.1 Stance Foot Slip

같은 주기의 후보 간 거리를 비교한다.

\[
d_{ij}
=
\left\|
{}^W\hat p_{B,i}
-
{}^W\hat p_{B,j}
\right\|_2
\]

다른 정상 후보가 0.05 m 안에 하나도 없으면 해당 후보를 즉시 평균에서 제외한다. 이 상태가 5회 연속, 25 ms 유지되면 해당 다리의 Slip을 Latch한다.

원시 유효 다리가 하나면 비교 없이 사용한다. 두 다리만 있고 서로 0.05 m보다 멀면 어느 다리가 정상인지 결정할 수 없으므로 둘 다 제외한다.

Slip Latch는 다리가 STANCE 또는 CONTACT 조건에서 벗어날 때 해제한다. 다음 Stance에서 Anchor를 새로 만들고 다시 판정한다.

\[
Slip\_Leg\_Mask
=
\sum_{i=1}^{6} Slip_i2^{i-1}
\]

Bit 0~5는 Leg 1~6이다. 현재 Slip Mask는 진단 출력이며 Safety Fault에는 연결하지 않는다.

### 5.2 최종 위치

Slip Reject를 통과한 다리 집합을 \(\mathcal S\), 개수를 \(N_S\)라 하면 다음처럼 단순 평균한다.

\[
{}^W\hat p_B
=
\frac{1}{N_S}
\sum_{i\in\mathcal S}
{}^W\hat p_{B,i}
\]

압력값 크기는 가중치로 사용하지 않는다. \(N_S=0\)이면 직전 유효 위치를 유지한다.

## 6. Position과 Heading PI

### 6.1 Position Reference

조종 모드의 몸체 x·y 속도를 목표 Heading으로 절대좌표계에 회전하고 적분한다.

\[
p_{B,ref}^{xy}[k]
=
p_{B,ref}^{xy}[k-1]
+
\begin{bmatrix}
\cos\psi_{ref} & -\sin\psi_{ref}\\
\sin\psi_{ref} & \cos\psi_{ref}
\end{bmatrix}
\begin{bmatrix}
v_{x,applied}\\
v_{y,applied}
\end{bmatrix}
T_s
\]

MANUAL 진입 또는 Reset 시에는 다음처럼 현재 위치에서 시작한다.

\[
p_{B,ref}^{xy}
\leftarrow
{}^W\hat p_B^{xy}
\]

요청 속도가 아니라 작업공간 검사를 통과한 적용 속도를 적분한다.

### 6.2 Position PI

\[
e_p
=
p_{B,ref}^{xy}
-
{}^W\hat p_B^{xy}
\]

\[
I_p[k]
=
\operatorname{clamp}
\left(
I_p[k-1]+e_p[k]T_s,
-I_{p,max},
I_{p,max}
\right)
\]

\[
v_{feedback}^{W}
=
\operatorname{sat}
\left(
K_{P,p}e_p+K_{I,p}I_p,
\pm v_{fb,max}
\right)
\]

현재 초기값은 다음과 같다.

| 항목 | 값 |
|---|---:|
| Position \(K_P\) | 1.0 |
| Position \(K_I\) | 0 |
| 적분 제한 | ±0.20 |
| 축별 Feedback 제한 | ±0.05 m/s |

Position PI는 유효 위치 후보가 하나 이상인 MANUAL에서만 활성화한다. z 위치와 CORRECTION에는 적용하지 않는다.

### 6.3 Heading PI

\[
e_\psi
=
\operatorname{wrap}_{\pi}
\left(
\psi_{ref}-\psi_{IMU}
\right)
\]

\[
\omega_{z,feedback}
=
\operatorname{sat}
\left(
K_{P,\psi}e_\psi+K_{I,\psi}I_\psi,
\pm15^\circ/\mathrm{s}
\right)
\]

현재 \(K_{P,\psi}=2.0\), \(K_{I,\psi}=0\), 적분 제한은 ±0.50이다.

PI 출력이 포화되거나 작업공간에서 거부된 방향으로 명령이 계속되면 적분을 중단한다. 비활성 모드에서는 출력과 적분항을 0으로 만든다.

## 7. 몸체 자세 PI와 오버레이

### 7.1 자세 오차

MANUAL에서는 Roll·Pitch만 사용한다.

\[
e_{posture}
=
\begin{bmatrix}
\operatorname{wrap}_{\pi}(\phi_{ref}-\phi)\\
\operatorname{wrap}_{\pi}(\theta_{ref}-\theta)\\
0
\end{bmatrix}
\]

CORRECTION 진입 시 Yaw 기준 \(\psi_{corr,0}\)를 현재 IMU Yaw로 저장한다.

\[
\psi_{corr,target}
=
\operatorname{wrap}_{\pi}
\left(
\psi_{corr,0}+\psi_{corr,ref}
\right)
\]

\[
e_{posture}
=
\begin{bmatrix}
0\\
0\\
\operatorname{wrap}_{\pi}(\psi_{corr,target}-\psi)
\end{bmatrix}
\]

### 7.2 단일 자세 PI

\[
I_R[k]
=
\operatorname{clamp}
\left(
I_R[k-1]+e_{posture}[k]T_s,
-0.50,
0.50
\right)
\]

\[
\omega_{posture}
=
\operatorname{sat}
\left(
K_{P,R}e_{posture}+K_{I,R}I_R,
\pm15^\circ/\mathrm{s}
\right)
\]

현재 Roll·Pitch·Yaw의 \(K_P\)는 2.0, \(K_I\)는 0이다. PI 출력은 자세 보정 각속도이며 별도의 내부 각속도 PI는 사용하지 않는다.

자세 명령은 다음처럼 적분한다.

\[
\eta_{cmd}[k+1]
=
\operatorname{clamp}
\left(
\eta_{cmd}[k]+\omega_{posture}[k]T_s,
\eta_{min},
\eta_{max}
\right)
\]

Roll·Pitch 범위는 ±45 deg, 보정 Yaw 범위는 ±10 deg이다.

### 7.3 자세 오버레이

자세 명령의 회전행렬은 Z-Y-X 순서를 사용한다.

\[
{}^RR_B
=
R_z(\psi_{cmd})R_y(\theta_{cmd})R_x(\phi_{cmd})
\]

명목 발끝 목표에 몸체 원점 기준 역회전을 적용한다.

\[
{}^Bp_{F_i}^{cmd}
=
({}^RR_B)^T
{}^Rp_{F_i}^{base}
\]

\(\{R\}\)과 \(\{B\}\)는 같은 원점을 사용하므로 평행이동은 추가하지 않는다. 자세 오버레이는 Tripod Phase, 보폭과 Swing 착지점을 바꾸지 않는다.

## 8. 최종 Body Twist

MANUAL에서는 사용자 Feedforward와 Position·Heading Feedback을 합한다.

\[
v_{gait}^{B}
=
v_{user}^{B}
+
{}^BR_Wv_{feedback}^{W}
\]

\[
\omega_{gait}^{B}
=
\begin{bmatrix}
0\\
0\\
\omega_{z,user}+\omega_{z,feedback}
\end{bmatrix}
\]

CORRECTION에서는 다음을 사용한다.

\[
v_{gait}^{B}
=
\begin{bmatrix}
v_{x,corr}\\
v_{y,corr}\\
v_{z,corr}
\end{bmatrix},
\qquad
\omega_{gait}^{B}=0
\]

Roll·Pitch와 보정 Yaw 자세 PI는 Body Twist에 합하지 않는다.

현재 명령 변화율 제한은 다음과 같다.

| 항목 | 값 |
|---|---:|
| 선가속도 제한 | 0.5 m/s² |
| Yaw 각가속도 제한 | 90 deg/s² |
| 5 ms당 선속도 변화 | 0.0025 m/s |
| 5 ms당 Yaw 속도 변화 | 0.45 deg/s |

최종 크기는 x·y 각각 0.28 m/s, z 0.05 m/s, Yaw 45 deg/s로 제한한다.

## 9. Tripod Gait Manager

Tripod A는 Leg 1·3·5, Tripod B는 Leg 2·4·6이다. 정상 보행에서는 두 그룹이 0.5 s마다 STANCE와 SWING을 교대한다.

\[
s_{phase}
=
\operatorname{clamp}
\left(
\frac{t-t_0}{0.5},
0,
1
\right)
\]

실제 적용 명령이 다음 중 하나를 만족할 때만 보행을 시작한다.

\[
|v_x|\ge0.005\ \mathrm{m/s}
\]

\[
|v_y|\ge0.005\ \mathrm{m/s}
\]

\[
|\omega_z|\ge1^\circ/\mathrm{s}
\]

보행 명령이 중립으로 돌아오면 진행 중인 Swing 다리를 착지시킨 뒤 Phase를 정지한다. 6개 발의 마지막 안전 STANCE 목표를 유지하며 0이나 기본 위치로 순간 변경하지 않는다.

Late Landing 중에는 시간 진행률이 1이어도 모든 Swing 발이 접촉하기 전까지 다음 Phase로 전환하지 않는다.

## 10. 발끝 궤적

### 10.1 Stance

몸체 기준 좌표계에서 지면에 고정된 Stance 발의 속도는 다음과 같다.

\[
\dot p_i=-v_B-\omega_B\times p_i
\]

200 Hz Forward Euler는 다음과 같다.

\[
p_i[k+1]
=
p_i[k]
+
\left(
-v_B[k]-\omega_B[k]\times p_i[k]
\right)T_s
\]

### 10.2 Swing 착지점

기본 발끝 위치를 \(p_{0,i}\), 예상 Stance 시간을 \(T_{stance}\)라 하면 착지점은 다음과 같다.

\[
p_{r,i}
=
p_{0,i}
+
\frac{T_{stance}}{2}
\left(
v_B+\omega_B\times p_{0,i}
\right)
\]

착지점과 전체 Swing 중간 궤적은 IK와 관절 범위를 만족해야 한다.

### 10.3 Bezier Curve

Swing 시작점 \(P_0\), 착지점 \(P_3\), Swing Height \(h\)에 대해 다음 제어점을 사용한다.

\[
P_1=P_0+
\begin{bmatrix}
0\\0\\\frac{4}{3}h
\end{bmatrix},
\qquad
P_2=P_3+
\begin{bmatrix}
0\\0\\\frac{4}{3}h
\end{bmatrix}
\]

\[
B(s)
=
(1-s)^3P_0
+
3(1-s)^2sP_1
+
3(1-s)s^2P_2
+
s^3P_3
\]

시간 진행률 \(\tau\)에는 Quintic Time Scaling을 적용한다.

\[
\tau
=
\operatorname{clamp}
\left(
\frac{t-t_s}{T_{swing}},
0,
1
\right)
\]

\[
s(\tau)
=
10\tau^3-15\tau^4+6\tau^5
\]

시작과 종료에서 속도와 가속도가 0이 되어 궤적 전환을 부드럽게 한다.

### 10.4 방사 오프셋

다리 바깥 방향 단위벡터는 다음과 같다.

\[
e_{r,i}
=
\begin{bmatrix}
\cos\alpha_i\\
\sin\alpha_i\\
0
\end{bmatrix}
\]

\[
\alpha_i=
[-45,-90,-135,45,90,135]^\circ
\]

진행률에 따른 방사 오프셋은 다음과 같다.

\[
r(s)=4r_{swing}s(1-s),
\qquad
r_{swing}=0.07\ \mathrm{m}
\]

최종 Swing 발끝 목표는 다음과 같다.

\[
p_i^{ref}
=
B(s)+r(s)e_{r,i}
\]

시작점과 착지점에서는 오프셋이 0이고 Swing 중간에서 0.07 m가 된다.

### 10.5 Swing Height

\[
h
=
\operatorname{clamp}
\left(
0.20+\Delta z_B,
0.15,
0.25
\right)
\ \mathrm{m}
\]

\(\Delta z_B\)는 몸체 기준점에 대한 몸체 원점의 z Offset이다. 현재 기본값은 0.20 m이며 0.25 m가 아니다.

## 11. 발 접촉과 착지 적응

### 11.1 접촉 Hysteresis

발별 압력값 \(F_i\)에 대해 다음 조건을 사용한다.

\[
F_{release}<F_{contact}
\]

연속 \(N_c\) Sample 동안 \(F_i>F_{contact}\)이면 CONTACT로 전환한다. 연속 \(N_r\) Sample 동안 \(F_i<F_{release}\)이면 CONTACT를 해제한다.

임계값과 Sample 수는 실측 후 발별 설정 테이블에 기록한다.

### 11.2 Early Landing

다음 조건을 모두 만족하면 Early Landing이다.

- Swing 진행률이 50% 이상이다.
- 발끝이 하강 중이다.
- CONTACT가 검출된다.

해당 주기의 현재 발끝 위치를 Stance 시작점으로 저장하고 즉시 STANCE로 전환한다. 다른 Swing 다리는 기존 궤적을 계속 수행한다.

### 11.3 Late Landing

Swing 종료 시 CONTACT가 없으면 Late Landing으로 전환한다. 발을 -Z 방향으로 내리면서 다리 안쪽으로 이동한다.

\[
v_{search}=0.20\ \mathrm{m/s}
\]

\[
v_{in}=0.8v_{search}=0.16\ \mathrm{m/s}
\]

\[
\dot p_{late,i}
=
\begin{bmatrix}
-v_{in}\cos\alpha_i\\
-v_{in}\sin\alpha_i\\
-v_{search}
\end{bmatrix}
\]

CONTACT가 검출되면 STANCE로 전환한다. 최대 탐색 거리와 최대 탐색 시간은 아직 정하지 않는다. 탐색 중 IK Invalid가 발생하면 Controller Fault를 Latch한다.

## 12. 좌표 변환과 IK

### 12.1 몸체에서 다리 좌표로 변환

i번 다리 원점의 몸체 좌표를 \({}^Bp_{L_i}\), 다리 회전행렬을 \({}^BR_{L_i}\)라 하면 다음과 같다.

\[
{}^{L_i}p_F^{ref}
=
({}^BR_{L_i})^T
\left(
{}^Bp_F^{cmd}-{}^Bp_{L_i}
\right)
\]

### 12.2 3DOF IK

현재 링크 길이는 다음과 같다.

| 링크 | 길이 |
|---|---:|
| \(L_1\) | 0.074 m |
| \(L_2\) | 0.121 m |
| \(L_3\) | 0.230 m |

다리 로컬 목표를 \([x,y,z]^T\)라 하면 다음을 계산한다.

\[
\theta_1=\operatorname{atan2}(y,x)
\]

\[
r=\sqrt{x^2+y^2},
\qquad
\rho=r-L_1
\]

\[
c_3
=
\frac{\rho^2+z^2-L_2^2-L_3^2}
{2L_2L_3}
\]

IK 해가 존재하려면 \(-1\le c_3\le1\)이어야 한다.

\[
\theta_3
=
\operatorname{atan2}
\left(
\sigma_3\sqrt{1-c_3^2},
c_3
\right)
\]

\[
\theta_2
=
\operatorname{atan2}(z,\rho)
-
\operatorname{atan2}
\left(
L_3\sin\theta_3,
L_2+L_3\cos\theta_3
\right)
\]

\(\sigma_3\)은 기구에 맞는 Knee 해를 선택한다. 실제 서보 각도는 관절별 방향과 Offset을 적용한다.

\[
\theta_{servo,ij}
=
s_{ij}\theta_{ij}
+
\theta_{offset,ij},
\qquad
s_{ij}\in\{-1,1\}
\]

관절별 \(s_{ij}\)와 \(\theta_{offset,ij}\)은 실기 캘리브레이션으로 정한다.

## 13. 동적 작업공간 제한

### 13.1 기본 유효 조건

평면 2-Link 거리는 다음 범위를 만족해야 한다.

\[
|L_2-L_3|+m_{ws}
\le
d
\le
L_2+L_3-m_{ws}
\]

\[
d=\sqrt{\rho^2+z^2},
\qquad
m_{ws}=0.0001\ \mathrm{m}
\]

IK 유효성은 거리, Cosine Law와 관절 범위를 함께 검사한다.

### 13.2 자세 명령

이번 주기의 Roll·Pitch·Yaw 후보를 6개 발에 함께 적용한다.

1. 자세 PI와 15 deg/s Rate Limit로 후보각을 만든다.
2. 후보 자세의 6개 발끝 위치를 계산한다.
3. 모든 다리가 유효하면 세 축 후보를 함께 적용한다.
4. 한 다리라도 유효하지 않으면 세 축을 모두 직전 적용값으로 유지한다.
5. 작업공간 안쪽으로 돌아오는 후보는 즉시 허용한다.

작은 200 Hz 연속 입력을 전제로 하므로 최대 허용각을 찾는 이분 탐색은 사용하지 않는다. 후보가 거부된 동안에는 자세 PI 적분항도 바깥 방향으로 누적하지 않는다.

### 13.3 보행 속도와 회전

한 Phase가 0.5 s이므로 한 축 0.28 m/s는 0.14 m의 발 이동량에 해당한다.

\[
l_{phase}
=
\sqrt{v_x^2+v_y^2}\times0.5
\]

x·y·Yaw가 동시에 입력되면 합성 방향과 회전 반경을 유지하도록 같은 비율과 공통 적용 여부를 사용한다. 다음 보행 궤적 전체가 유효할 때만 새 값을 적용하며 Phase 중간에는 적용값을 바꾸지 않는다.

x·y를 각각 0.28 m/s로 동시에 고정하면 합성 속도가 한 축 제한을 넘을 수 있다. 실제 적용값은 합성 작업공간 검사를 통과한 범위로 제한하며 최대값 시험도 축별로 수행한다.

### 13.4 보정 위치

\[
p_{candidate}[k]
=
p_{applied}[k-1]
+
v_{request}[k]T_s
\]

6개 다리가 모두 유효하면 후보 위치를 적용한다. 불가능하면 바깥 방향 속도를 0으로 만들고 적분 상태를 갱신하지 않는다. 안쪽 복귀 속도는 허용한다.

### 13.5 최종 발끝 보호

동적 제한 이후 각 IK 입력에도 0.0001 m 여유의 발끝 제한을 적용한다. 이 제한은 수치 오차 보호용이다. 정상 동작에서 큰 발끝 수정이 발생하면 상위 동적 제한 오류로 판단한다.

## 14. 전환 연속성

시간 기반 모드 전환 보간은 사용하지 않는다. 다음 상태 초기화로 출력 연속성을 확보한다.

- 모드 변경 전에 새 입력 필터 상태를 초기화한다.
- MANUAL 진입 시 Position Reference를 현재 추정 위치에 맞춘다.
- MANUAL 진입과 SA 전환 시 Heading Reference를 현재 IMU Yaw에 맞춘다.
- 새 PI 모드 진입 시 적분항을 0으로 초기화한다.
- 자세·보정 후보가 불가능하면 직전 유효 명령을 유지한다.
- 보행 정지는 Swing 착지를 끝낸 뒤 마지막 STANCE 목표를 유지한다.
- 발끝 목표를 0이나 기본 위치로 순간 초기화하지 않는다.

## 15. Safety

### 15.1 현재 구현

Safety는 다음 두 Fault를 출력한다.

\[
Rollover\_Fault
=
(|Roll|\ge80^\circ)
\lor
(|Pitch|\ge80^\circ)
\]

단, IMU 값이 유한하지 않으면 Rollover 계산 대신 Controller Fault로 처리한다.

\[
Controller\_Fault
=
IMU\_Invalid
\lor
IK\_Invalid
\]

IK Invalid는 6개 다리 중 하나라도 IK Valid가 0이면 참이다.

두 Fault는 한 번 발생하면 영구 Latch한다. Safety Reset은 없으며 자동 Recovery도 수행하지 않는다. Fault가 Latch되면 FAULT 상태를 거쳐 Kill을 활성화하고 6개 서보 전원 릴레이를 모두 끈다.

### 15.2 상시 출력 보호

다음 제한은 Safety Latch와 별개로 항상 적용한다.

- 발끝 작업공간
- 관절각 -135~135 deg
- 관절 명령 속도 315.8 deg/s
- PWM 500~2500 us

### 15.3 후속 후보

다음 항목은 현재 Safety에 포함하지 않는다.

- Safe Support Polygon
- Joint Jump Fault
- 센서별 Timeout Fault
- Stance Foot Slip의 Fault 승격

Late Landing 최대 거리와 최대 시간도 아직 정하지 않는다.

## 16. Servo Output

관절각을 -135~135 deg에서 500~2500 us로 선형 변환한다.

\[
PWM_{ij}
=
1500
+
\frac{1000}{135}
\theta_{ij}^{cmd}
\]

최종 PWM은 500~2500 us로 제한한다.

IK와 관절 범위 제한 뒤에 관절별 Rate Limiter를 적용한다.

\[
\Delta\theta_{max}
=
315.8^\circ/\mathrm{s}\times0.005\ \mathrm{s}
=
1.579^\circ
\]

\[
\theta_{ij}^{cmd}[k]
=
\theta_{ij}^{cmd}[k-1]
+
\operatorname{clamp}
\left(
\theta_{ij}^{ref}[k]-\theta_{ij}^{cmd}[k-1],
-1.579^\circ,
1.579^\circ
\right)
\]

Rate Limit 뒤에 관절별 방향·중립점 보정과 PWM 변환을 수행한다. 실기에서는 관절 ADC로 실제 추종 속도를 측정해 필요하면 제한값을 낮춘다.

## 17. 5 ms 실행 순서

1. IMU, 관절 ADC와 압력센서 Snapshot을 갱신한다.
2. CRSF 명령과 연결 상태를 갱신한다.
3. 현재 IMU와 직전 IK 결과로 Safety를 평가한다.
4. 상위 상태와 동작 허가를 결정한다.
5. 사용자 입력을 정규화하고 필터링한다.
6. 위치·Heading·자세 Reference를 갱신한다.
7. FK 기반 몸체 위치와 Slip 상태를 계산한다.
8. Position·Heading·자세 PI를 계산한다.
9. 적용 가능한 Body Twist와 자세 후보를 결정한다.
10. Tripod 상태와 Stance/Swing 궤적을 갱신한다.
11. Early/Late Landing을 적용한다.
12. 몸체 자세 오버레이와 다리 좌표 변환을 수행한다.
13. 6개 다리 IK와 유효성을 계산한다.
14. 관절 범위와 Rate Limit를 적용한다.
15. Fault가 없으면 PWM을 갱신하고, Fault 또는 Kill이면 릴레이를 끈다.
16. 남은 시간 동안 다음 5 ms Tick을 기다린다.

GPS, LoRa와 Jetson SPI는 필요한 주기에 따라 백그라운드에서 처리한다.

## 18. 튜닝과 실기 검증

다음 순서로 진행한다.

1. 서보 전원을 끄고 좌표변환, FK와 IK 수치를 검증한다.
2. 관절 ADC의 최소·영점·최대값과 방향을 측정한다.
3. 압력센서의 접촉·해제 임계값과 Sample 수를 정한다.
4. 서보별 방향, 중립 Pulse와 안전 Pulse 범위를 측정한다.
5. 낮은 속도로 Stance와 Swing 궤적을 검증한다.
6. Position \(K_P\)를 먼저 조정하고 필요할 때만 작은 \(K_I\)를 추가한다.
7. Heading과 몸체 자세 PI도 P부터 조정한다.
8. Early/Late Landing과 Stance Foot Slip을 검증한다.
9. 동적 작업공간 제한과 모드 전환 연속성을 검증한다.
10. IK Invalid와 80 deg 전복 입력에서 Safety Latch와 릴레이 차단을 확인한다.
11. 실기 속도를 단계적으로 0.28 m/s까지 올린다.

## 19. 현재 파라미터 요약

| 구분 | 항목 | 값 |
|---|---|---:|
| 제어 | 주기 | 0.005 s |
| RC | Throttle Dead Zone | ±20 raw |
| RC | Stick Dead Zone | ±50 raw |
| RC | LPF Cutoff | 5 Hz |
| RC | 시험·RL 변화율 | 5000 raw/s |
| 조종 | x·y 최대 속도 | 각 ±0.28 m/s |
| 조종 | Yaw 최대 속도 | ±45 deg/s |
| 조종 | Roll·Pitch 최대각 | ±45 deg |
| 보정 | x·y·z 최대 속도 | 각 ±0.05 m/s |
| 보정 | Yaw 최대각 | ±10 deg |
| PI | Position \(K_P,K_I\) | 1.0, 0 |
| PI | Heading \(K_P,K_I\) | 2.0, 0 |
| PI | Posture \(K_P,K_I\) | 2.0, 0 |
| Feedback | Position 최대 | ±0.05 m/s |
| Feedback | Heading 최대 | ±15 deg/s |
| Posture | 명령 속도 제한 | ±15 deg/s |
| Gait | Phase 시간 | 0.5 s |
| Swing | 기본 높이 | 0.20 m |
| Swing | 높이 범위 | 0.15~0.25 m |
| Swing | 방사 오프셋 | 0.07 m |
| Early | 판정 시작 | Swing 50% |
| Late | 하강 속도 | 0.20 m/s |
| Late | 안쪽 속도 | 0.16 m/s |
| Estimator | Slip 거리 | 0.05 m |
| Estimator | Slip 확정 | 5 Sample, 25 ms |
| Workspace | 여유 | 0.0001 m |
| Joint | 범위 | -135~135 deg |
| Joint | 명령 속도 | 315.8 deg/s |
| Joint | 5 ms 변화량 | 1.579 deg |
| Safety | 전복 기준 | Roll 또는 Pitch 80 deg |
