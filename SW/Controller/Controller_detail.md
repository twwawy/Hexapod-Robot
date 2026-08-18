# 6족 로봇 보행 제어기 상세 설계

본 문서는 `Controller_Architecture.md`에서 정의한 전체 제어 구조를 실제 구현 가능한 수준의 수식과 이산시간 제어식으로 구체화한다.

기본 구조는 다음을 유지한다.

- 전체 제어 주기: **200 Hz**
- 기본 보행: **Tripod Gait**
- 사용자 명령 + Body Position Feedback + Gait Heading Yaw Feedback → **Final Gait Body Twist**
- 조종 Roll·Pitch + 보정 Yaw 단일 자세 PI → **IK 직전 몸체 자세 오버레이**
- Stance 다리: Final Gait Body Twist의 반대 운동으로 **PULL 궤적** 생성
- Swing 다리: 착지 목표점 + **3차 Bezier Curve + 방사 방향 오프셋**
- 발끝 압력센서: **Early / Late Landing**
- 발끝 목표 위치 → 좌표계 변환 → **3DOF IK**
- 최종 관절각에 안전 제한 적용 후 Servo Output

---

# 1. 기호 및 좌표계

## 1.1 좌표계

좌표계 정의는 좌표축 README 및 `Controller_Architecture.md`의 정의를 따른다.

- $\{B\}$: 현재 몸체 원점 좌표계
- $\{W\}$: 지면에 고정된 절대 좌표계
- $\{R\}$: 몸체 기준점 좌표계
- $\{L_i\}$: i번 다리 좌표계
- $O_W$: 최초 관절각 0° 착지 상태에서 정의한 절대 원점
- $O_B$: 현재 몸체 원점
- $O_R$: 몸체 기준점
- $F_i$: i번 다리 발끝

몸체 좌표계 $\{B\}$의 축은

$$
+x_B:\text{전진},\qquad
+y_B:\text{왼쪽},\qquad
+z_B:\text{위쪽}
$$

으로 정의한다.

절대 좌표계 $\{W\}$는 로봇의 이동과 무관하게 지면에 고정한다. 최초 관절각 0° 착지 상태에서 $O_W=O_B$로 초기화하며, 기본 자세로 일어서면 ${}^Wp_B=[0,0,h_{stand}]^T$가 된다. 몸체 좌표계 $\{B\}$와 몸체 기준 좌표계 $\{R\}$는 로봇의 명목 이동을 따라간다. Body Position Estimator는 절대좌표계의 x·y·z 위치를 추정하고, Position PI의 x·y 목표·측정·오차는 절대 좌표계 $\{W\}$에서 계산한다.

## 1.2 벡터

몸체 선속도와 각속도는

$$
v_B=
\begin{bmatrix}
v_x\\v_y\\v_z
\end{bmatrix},
\qquad
\omega_B=
\begin{bmatrix}
\omega_x\\\omega_y\\\omega_z
\end{bmatrix}
$$

로 정의한다.

Body Twist는

$$
\xi_B=
\begin{bmatrix}
v_B\\
\omega_B
\end{bmatrix}
=
\begin{bmatrix}
v_x\\v_y\\v_z\\
\omega_x\\\omega_y\\\omega_z
\end{bmatrix}
$$

이다.

## 1.3 Skew-Symmetric Matrix

벡터

$$
a=
\begin{bmatrix}
a_x\\a_y\\a_z
\end{bmatrix}
$$

에 대해

$$
[a]_\times=
\begin{bmatrix}
0&-a_z&a_y\\
a_z&0&-a_x\\
-a_y&a_x&0
\end{bmatrix}
$$

로 정의하면

$$
a\times b=[a]_\times b
$$

로 표현할 수 있다.

---

# 2. 제어 주기

전체 제어 루프는 200 Hz로 동작한다.

$$
f_s=200\text{ Hz}
$$

$$
\boxed{\Delta t=T_s=0.005\text{ s}}
$$

따라서 한 번의 전체 제어 계산은 다음 제어 주기가 시작되기 전인 5 ms 이내에 완료되어야 한다.

$$
\boxed{
T_{exec}<5\text{ ms}
}
$$

실제 구현에서는 연산 여유를 확보하기 위해 실행 시간을 측정하고, 가능한 한 5 ms에 충분한 여유를 두도록 한다.

모든 PI 적분, 발끝 궤적 적분, 관절 각속도 제한은 동일한 $0.005\,\mathrm{s}$의 제어 주기를 기준으로 계산한다.

제어 주기 번호를 $k$라 하면 현재 시각은

$$
t_k=kT_s
$$

이다.

---

# 3. 조종기 입력 전처리

조종기에서 들어오는 정규화 입력을

$$
u_j\in[-1,1]
$$

로 정의한다.

현재 조종 모드의 기본 입력은

$$
u_T:\text{Throttle}
$$

$$
u_R:\text{Roll}
$$

$$
u_P:\text{Pitch}
$$

$$
u_Y:\text{Yaw}
$$

이다.

---

## 3.1 Dead Zone

조이스틱 중앙 노이즈를 제거하기 위해 Dead Zone을 적용한다.

Dead Zone 크기를 $\delta$라 하면

$$
D(u)=
\begin{cases}
0,
&
|u|\le\delta
\\[6pt]
\operatorname{sgn}(u)
\dfrac{|u|-\delta}{1-\delta},
&
|u|>\delta
\end{cases}
$$

로 정의한다.

이 식을 사용하면 Dead Zone 외부의 입력 범위가 다시 $[-1,1]$로 정규화된다.

각 입력에 대해

$$
u_{T,d}=D(u_T)
$$

$$
u_{R,d}=D(u_R)
$$

$$
u_{P,d}=D(u_P)
$$

$$
u_{Y,d}=D(u_Y)
$$

를 계산한다.

---

## 3.2 Low Pass Filter

Dead Zone을 통과한 조종기 입력에는 1차 Low Pass Filter를 적용한다.

연속시간 1차 LPF

$$
G(s)=\frac{\omega_c}{s+\omega_c}
$$

를 이산화하여

$$
u_f[k]
=
\alpha u_f[k-1]
+
(1-\alpha)u_d[k]
$$

로 사용한다.

여기서

$$
\alpha=e^{-2\pi f_cT_s}
$$

이고 $f_c$는 Cutoff Frequency이다.

따라서 각 채널은

$$
u_{T,f},\quad
u_{R,f},\quad
u_{P,f},\quad
u_{Y,f}
$$

로 변환된다.

---

## 3.3 Control Priority Manager와 Drone Controller

조종기 입력 처리와 동작 모드 생성은 다음 순서를 따른다.

```text
USER 또는 실제 RC 입력
        ↓
ControlPriorityManager
        ↓
DroneController
        ↓
기능별 Enable, 진행률 및 사용자 명령
```

`ControlPriorityManager`는 조종기 스위치, 서기·착지 완료 신호와 Fault 신호를 이용하여 현재 허용할 `Active_Mode`를 먼저 결정한다. `DroneController`는 전달받은 `Active_Mode`에 해당하는 기능만 갱신한다. 이 순서를 사용하면 우선순위에서 차단된 서기·착지 또는 보행 진행률이 내부에서 미리 진행되는 것을 방지할 수 있다.

상위 제어 상태의 우선순위는 다음과 같다.

$$
\boxed{
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
READY(MANUAL/CORRECTION)
}
$$

`Motion_Armed`는 READY 진입 직후 남아 있는 조종기 입력으로 갑작스러운 동작이 발생하는 것을 방지하는 내부 허가 상태이다. READY가 아니면 즉시 해제하고, READY 진입 후 Throttle·Yaw·Roll·Pitch가 연속 0.2초 동안 중립 범위에 있을 때 활성화한다. 활성화 전에는 수동 조종 및 보정 명령을 출력하지 않는다.

수동 조종 모드에서는 실제 이동 명령이 다음 조건 중 하나를 만족할 때만 Tripod 보행을 활성화한다.

$$
\boxed{
Tripod\_Enable
=
\left(
|v_{x,user}|\ge V_{gait,th}
\right)
\lor
\left(
|\omega_{z,user}|\ge \Omega_{gait,th}
\right)
}
$$

따라서 READY 또는 MANUAL 상태에 진입했다는 이유만으로 Tripod 보행을 시작하지 않는다.

---

# 4. 사용자 Body Command

`Controller_Architecture.md`의 정의에 따라 현재 조종 모드에서는

- Throttle → 전진·후진 속도
- Roll → 몸체 Roll 목표각
- Pitch → 몸체 Pitch 목표각
- Yaw → 몸체 Yaw 각속도

를 명령한다.

최대 명령값을

$$
V_{x,\max},
\quad
\phi_{\max},
\quad
\theta_{\max},
\quad
\Omega_{z,\max}
$$

라 하면

$$
v_{x,user}
=
V_{x,\max}u_{T,f}
$$

$$
\phi_{ref}
=
\phi_{\max}u_{R,f}
$$

$$
\theta_{ref}
=
\theta_{\max}u_{P,f}
$$

$$
\omega_{z,user}
=
\Omega_{z,\max}u_{Y,f}
$$

로 변환한다.

현재 조종 모드에서 별도의 측방 및 높이 속도 명령을 사용하지 않는다면

$$
v_{y,user}=0
$$

$$
v_{z,user}=0
$$

으로 둔다.

따라서

$$
\boxed{
\xi_{user}
=
\begin{bmatrix}
v_{x,user}\\
0\\
0\\
0\\
0\\
\omega_{z,user}
\end{bmatrix}
}
$$

이다.

보정 모드의 Yaw 입력은 조종 모드의 $\omega_{z,user}$와 구분하여 yaw 보정 목표각으로 사용한다. 최대 yaw 보정각을 $\psi_{corr,max}$라 하면

$$
\boxed{
\psi_{corr,ref}[k]
=
\psi_{corr,max}u_{Y,f}[k]
}
$$

로 직접 변환한다. $\psi_{corr,ref}$는 각속도 명령이 아니므로 시간 적분하지 않는다. 보정 모드의 yaw 조이스틱이 중앙으로 복귀하면 $\psi_{corr,ref}=0$으로 복귀하며, 필요한 실제 보정 각속도는 이후 자세 오차 제어에서 생성한다.

---

# 5. 몸체 자세 목표값 생성

Roll과 Pitch 조이스틱 입력은 **목표각 명령**으로 사용한다. 필터링된 입력을 각 축의 최대 목표각에 직접 대응시킨다.

$$
\phi_{ref}[k]
=
\phi_{\max}u_{R,f}[k]
$$

$$
\theta_{ref}[k]
=
\theta_{\max}u_{P,f}[k]
$$

조종 모드의 Yaw 조이스틱 입력만 **각속도 명령**으로 사용한다. Heading Reference는

$$
\psi_{ref}[k+1]
=
\operatorname{wrap}_{\pi}
\left(
\psi_{ref}[k]
+
\omega_{z,user}[k]T_s
\right)
$$

로 갱신한다.

Roll과 Pitch의 최대 허용 자세를 각각

$$
\phi_{\max},
\qquad
\theta_{\max}
$$

라 하면

$$
\phi_{ref}
=
\operatorname{clamp}
(
\phi_{ref},
-\phi_{\max},
\phi_{\max}
)
$$

$$
\theta_{ref}
=
\operatorname{clamp}
(
\theta_{ref},
-\theta_{\max},
\theta_{\max}
)
$$

를 적용한다.

> 조종 모드에서 Roll·Pitch 조이스틱이 중앙으로 복귀하면 목표각은 0으로 복귀한다. Yaw 조이스틱이 중앙으로 복귀하면 Yaw 각속도 명령만 0이 되고, 목표 Heading은 그 순간의 값을 유지한다.

조종 모드의 Yaw 축은 **Heading Hold를 기본으로 사용한다.**

조종 모드의 Yaw 조이스틱 입력은 목표 Yaw 각도 자체가 아니라 목표 Yaw 각속도 $\omega_{z,user}$를 명령한다. 따라서 조종 모드에서 Yaw 조이스틱을 움직이는 동안에는 목표 Heading을 다음과 같이 적분하여 갱신한다. 보정 모드의 $\psi_{corr,ref}$에는 이 적분식을 적용하지 않는다.

$$
\psi_{ref}[k+1]
=
\operatorname{wrap}_{\pi}
\left(
\psi_{ref}[k]
+
\omega_{z,user}[k]T_s
\right)
$$

조종 모드에서 Yaw 조이스틱이 중앙으로 복귀하면

$$
\omega_{z,user}=0
$$

이므로

$$
\psi_{ref}[k+1]
=
\psi_{ref}[k]
$$

가 되어 마지막 목표 Heading을 유지한다.

이후 IMU에서 측정한 현재 Yaw와 목표 Yaw의 오차를

$$
e_{\psi}
=
\operatorname{wrap}_{\pi}
\left(
\psi_{ref}-\psi
\right)
$$

로 계산하고, Yaw PI를 이용하여 보정 각속도를 생성한다.

$$
\omega_{z,feedback}
=
K_{P,\psi}e_{\psi}
+
K_{I,\psi}I_{\psi}
$$

따라서 조이스틱이 중앙에 있어도 보행 중 좌우 다리의 추진력 차이, 지면 마찰 차이 또는 외력으로 몸체의 Yaw가 틀어지면 Yaw Feedback이 목표 Heading으로 복원한다.

제어기 최초 활성화, MANUAL 모드 진입 또는 Heading Hold 재활성화 시에는 이전 목표값 때문에 갑작스러운 회전 명령이 발생하지 않도록 현재 Yaw를 목표값으로 초기화한다.

$$
\boxed{
\psi_{ref}
\leftarrow
\psi
}
$$

---

# 6. 몸체 상태 추정

몸체 상태 Feedback에 필요한 상태는 크게 $\hat p_B$와 $\hat R_B$이다.

- $\hat p_B$: 절대 좌표계에 대한 몸체 원점의 추정 위치
- $\hat R_B$: 절대 좌표계에 대한 IMU 기반 몸체 추정 자세

---

# 7. Body Position 추정

## 7.1 Stance 발끝의 순기구학

i번 다리의 실제 관절각을

$$
q_i=
\begin{bmatrix}
\theta_{i1}\\
\theta_{i2}\\
\theta_{i3}
\end{bmatrix}
$$

라 한다.

각 다리의 Forward Kinematics를 이용하여 현재 몸체 좌표계에서의 발끝 위치를 계산한다.

$$
{}^Bp_{F_i}^{meas}
=
FK_i(q_i)
$$

Stance 상태이며 압력센서로 접촉이 확인된 다리만 위치 추정에 사용한다.

---

## 7.2 Stance 발 고정 조건

지면에 미끄럼이 없다고 가정하면 Stance 발끝은 기준 좌표계에서 고정되어 있다.

$$
{}^W\dot p_{F_i}\approx0
$$

몸체 자세 회전행렬을

$$
{}^WR_B
$$

라 하면 발끝의 위치 관계는

$$
{}^Wp_{F_i}
=
{}^Wp_B
+
{}^WR_B
{}^Bp_{F_i}
$$

이다.

따라서 i번 Stance 발을 이용한 몸체 위치 추정값은

$$
\boxed{
{}^W\hat p_{B,i}
=
{}^Wp_{F_i}^{anchor}
-
{}^WR_B
{}^Bp_{F_i}^{meas}
}
$$

로 계산할 수 있다.

여기서

$$
{}^Wp_{F_i}^{anchor}
$$

는 해당 발이 Stance를 시작할 때 저장한 지면 고정 발끝 위치이다.

---

## 7.3 Stance Anchor 초기화

Swing → Stance 전환 순간을 $k_s$라 하면

$$
{}^Wp_{F_i}^{anchor}
=
{}^W\hat p_B[k_s]
+
{}^WR_B[k_s]
{}^Bp_{F_i}^{meas}[k_s]
$$

로 저장한다.

이후 해당 발이 Stance인 동안 Anchor는 변경하지 않는다.

---

## 7.4 여러 Stance 다리의 단순 평균

Tripod Gait에서는 일반적으로 세 개의 Stance 다리가 지면을 지지한다.

각 Stance 다리의 관절각과 Forward Kinematics를 이용하면 각 다리로부터 몸체 위치 추정값을 각각 계산할 수 있다.

$$
{}^W\hat p_{B,i}
=
{}^Wp_{F_i}^{anchor}
-
{}^WR_B
{}^Bp_{F_i}^{meas}
$$

실제 로봇에서는 관절각 오차, 링크 변형, 발 미끄러짐 등의 영향으로 각 다리에서 계산한 몸체 위치가 완전히 동일하지 않을 수 있다.

따라서 현재 **STANCE 상태이면서 압력센서에서 CONTACT가 확인된 다리만** 위치 추정에 사용한다.

유효 Stance 다리 집합을 $\mathcal S$라 하고, 유효 다리의 개수를 $N_S=|\mathcal S|$라 하면 최종 몸체 위치는 각 다리의 추정값을 단순 평균하여 계산한다.

$$
\boxed{
{}^W\hat p_B
=
\frac{1}{N_S}
\sum_{i\in\mathcal S}
{}^W\hat p_{B,i}
}
$$

정상적인 Tripod Gait에서 세 Stance 다리가 모두 접촉하고 있다면 $N_S=3$이다.

예를 들어 Tripod A가 Stance인 경우

$$
\boxed{
{}^W\hat p_B
=
\frac{
{}^W\hat p_{B,1}
+
{}^W\hat p_{B,3}
+
{}^W\hat p_{B,5}
}{3}
}
$$

이고, Tripod B가 Stance인 경우

$$
\boxed{
{}^W\hat p_B
=
\frac{
{}^W\hat p_{B,2}
+
{}^W\hat p_{B,4}
+
{}^W\hat p_{B,6}
}{3}
}
$$

으로 계산한다.

압력센서 값의 크기는 가중치로 사용하지 않고, 각 발이 지면에 접촉했는지를 판단하는 **CONTACT / NO CONTACT 판정에만 사용한다.**

유효 Stance 다리가 하나도 없는 경우에는

$$
N_S=0
$$

이므로 몸체 위치를 새로 계산하지 않고 직전 유효 추정값을 유지하며, 정상적인 지지 상태가 아니므로 별도의 Fault 또는 Recovery 조건으로 처리한다.


## 7.5 Foot Slip Reject

특정 다리의 위치 추정값이 유효 Stance 다리의 단순 평균에서 지나치게 벗어나면 미끄러짐 또는 FK 이상으로 판단할 수 있다.

$$
r_i
=
\left\|
{}^W\hat p_{B,i}
-
{}^W\hat p_B
\right\|
$$

$$
r_i>r_{slip,max}
$$

이면 해당 다리를 현재 제어 주기의 유효 Stance 다리 집합 $\mathcal S$에서 제외하고, 나머지 유효 다리만 다시 단순 평균한다.

---

# 8. Body Position PI Controller

Body Position Estimator는 절대좌표계의 x·y·z 위치를 계산하지만, 현재 Position PI는 조종 모드의 Throttle 명령에 대해 x·y 위치만 제어한다. z 위치는 추정과 검증에 사용하며 Position PI에는 입력하지 않는다.

## 8.1 Position Reference Generator

조종 모드의 전진·후진 선속도 명령을

$$
v_{x,user}[k]
$$

라 하고 현재 Yaw Reference를 $\psi_{ref}[k]$라 하면, 절대좌표계의 목표 x·y 위치는

$$
\boxed{
p_{B,ref}^{xy}[k]
=
p_{B,ref}^{xy}[k-1]
+
\begin{bmatrix}
\cos\psi_{ref}[k]\\
\sin\psi_{ref}[k]
\end{bmatrix}
v_{x,user}[k]T_s
}
$$

로 갱신한다.

Throttle 선속도는 몸체 전진축 명령이므로 Yaw Reference를 이용하여 절대좌표계의 x·y 속도로 변환한다.

Position Reference Generator를 활성화할 때는 명령 점프를 방지하기 위해

$$
\boxed{
p_{B,ref}^{xy}[k_0]
=
{}^W\hat p_B^{xy}[k_0]
}
$$

로 초기화한다.

$v_{x,user}=0$이면 $p_{B,ref}^{xy}$는 현재 목표 위치에 고정되므로 조종 모드에서 정지 위치 유지가 가능하다. 절대 원점 유지 시험에서만

$$
p_{B,ref}^{xy}
=
\begin{bmatrix}
0\\0
\end{bmatrix}
$$

을 사용할 수 있다. 일반 보행 중에 $p_{B,ref}^{xy}=0$을 고정하면 Position PI가 사용자의 이동 명령을 반대로 보정하므로 사용하지 않는다.

## 8.2 Position Error

PI에 입력하는 위치는 개별 다리의 FK 발끝 좌표가 아니라, FK와 Stance Anchor를 이용해 계산한 Body Position Estimate의 x·y 성분 ${}^W\hat p_B^{xy}$이다.

위치 오차는

$$
\boxed{
e_p^{xy}[k]
=
p_{B,ref}^{xy}[k]
-
{}^W\hat p_B^{xy}[k]
}
$$

이다.

축별 PI Gain을 대각행렬로 정의한다.

$$
K_{P,p}
=
\operatorname{diag}
(
K_{Px},K_{Py}
)
$$

$$
K_{I,p}
=
\operatorname{diag}
(
K_{Ix},K_{Iy}
)
$$


---

## 8.3 적분항

$$
I_p[k]
=
I_p[k-1]
+
e_p^{xy}[k]T_s
$$

적분 포화를 막기 위해

$$
I_p[k]
=
\operatorname{clamp}
(
I_p[k],
I_{p,min},
I_{p,max}
)
$$

를 적용한다.

---


## 8.4 PI 출력

포화 전 위치 보정 속도는

$$
v_{feedback}^{raw}
=
K_{P,p}e_p^{xy}
+
K_{I,p}I_p
$$

이다.

축별 최대 보정 속도를

$$
v_{fb,max}
=
\begin{bmatrix}
v_{x,fb,max}\\
v_{y,fb,max}
\end{bmatrix}
$$

라 하면

$$
\boxed{
v_{feedback}
=
\operatorname{sat}
\left(
v_{feedback}^{raw},
-v_{fb,max},
v_{fb,max}
\right)
}
$$

를 적용한다.

$v_{feedback}$은 절대좌표계의 x·y 보정 속도이다. 발끝 궤적에 입력하기 전에 현재 몸체 Yaw를 이용하여 몸체 좌표계 속도로 변환하고, z축 보정 속도는 0으로 둔다.

---

## 8.5 모드별 활성 조건

Position PI는 `Motion_Armed`가 활성화된 조종 모드에서만 사용한다. 보정 모드의 x·y·z 이동속도와 암 모드 입력에는 Position PI를 적용하지 않는다.

Position PI는 `Tripod_Enable`과 독립적으로 동작한다. `Body_Control_Enable=1`, `Tripod_Enable=0`인 조종 모드에서는 6개 발을 모두 STANCE로 유지한 상태에서 x·y 위치 오차를 보정한다.

LANDED, STANDING, LANDING, CORRECTION, ARM, FAULT, KILL 상태에서는 Position PI 출력을 0으로 만들고 적분항을 누적하지 않는다. 조종 모드에 새로 진입할 때는

$$
p_{B,ref}^{xy}[k_0]
=
{}^W\hat p_B^{xy}[k_0]
$$

로 목표 위치를 현재 추정 위치에 맞추고 적분항을 0으로 초기화하여 명령 점프를 방지한다.

---

## 8.6 Anti-Windup

출력 포화 시 적분항이 계속 증가하지 않도록 Back-Calculation Anti-Windup을 적용할 수 있다.

$$
I_p[k]
=
I_p[k]
+
K_{aw,p}
\left(
v_{feedback}
-
v_{feedback}^{raw}
\right)T_s
$$

초기 구현에서는 Conditional Integration 방식도 사용할 수 있다.

즉, 출력이 포화되었고 현재 오차가 포화를 더 증가시키는 방향이면 적분을 정지한다.

---

# 9. Body Attitude 추정

Body Attitude PI Controller에서는 자세 상태를 **Roll, Pitch, Yaw Euler Angle**로 표현한다.

9축 IMU와 자세 추정 필터로

$$
\phi,\qquad
\theta,\qquad
\psi
$$

를 계산한다.

Euler Angle 벡터를

$$
\eta=
\begin{bmatrix}
\phi\\\theta\\\psi
\end{bmatrix}
$$

로 정의한다.

---

# 10. Body Attitude PI Controller

몸체 자세 제어는 **단일 자세 PI**를 사용한다. 내부 각속도 PI를 추가한 이중 PI 구조는 사용하지 않는다. 조종 모드에서는 Roll·Pitch를 제어하고, 보정 모드에서는 Yaw만 제어한다. 조종 모드 Yaw는 이 자세 PI가 아니라 5장의 Gait Heading Yaw PI에서 보행 방향을 제어한다.

## 10.1 Euler Angle 자세 오차

본 로봇은 $|Roll|\ge80^\circ$ 또는 $|Pitch|\ge80^\circ$인 경우를 안전장치에서 전복 상태로 처리하므로, 정상 자세 제어에서는 Pitch가 짐벌락이 발생하는 $\pm90^\circ$에 도달하지 않는다. 따라서 Body Attitude PI의 자세 오차는 Roll, Pitch, Yaw Euler Angle을 직접 사용한다.


조종 모드의 자세 오차는 Roll과 Pitch만 사용하여

$$
e_{posture}
=
\begin{bmatrix}
e_\phi\\
e_\theta\\
0
\end{bmatrix}
$$

$$
e_\phi
=
\operatorname{wrap}_{\pi}
(
\phi_{ref}-\phi
)
$$

$$
e_\theta
=
\operatorname{wrap}_{\pi}
(
\theta_{ref}-\theta
)
$$

로 계산한다.

보정 모드에서는 Roll·Pitch 성분을 0으로 두고 Yaw 자세 PI만 사용한다. 보정 모드 진입 시의 기준 Heading을 $\psi_{corr,0}$, 조종기에서 직접 생성한 상대 Yaw 보정 목표각을 $\psi_{corr,ref}$라 하면 절대 목표각은

$$
\psi_{corr,target}
=
\operatorname{wrap}_{\pi}
\left(
\psi_{corr,0}
+
\psi_{corr,ref}
\right)
$$

이고 Yaw 오차는

$$
\boxed{
e_{\psi,corr}
=
\operatorname{wrap}_{\pi}
\left(
\psi_{corr,target}-\psi
\right)
}
$$

로 계산한다.

따라서 보정 모드의 자세 오차 벡터는

$$
e_{posture}
=
\begin{bmatrix}
0\\0\\e_{\psi,corr}
\end{bmatrix}
$$

로 둔다.

---


## 10.2 단일 자세 PI

Gain 행렬을

$$
K_{P,R}
=
\operatorname{diag}
(
K_{P\phi},
K_{P\theta},
K_{P\psi}
)
$$

$$
K_{I,R}
=
\operatorname{diag}
(
K_{I\phi},
K_{I\theta},
K_{I\psi}
)
$$


로 둔다.

적분항은

$$
I_R[k]
=
I_R[k-1]
+
e_{posture}[k]T_s
$$

이다.

포화 전 출력은 몸체 각속도 보정 명령이며

$$
\omega_{posture}^{raw}
=
K_{P,R}e_{posture}
+
K_{I,R}I_R
$$

이다.

최대 자세 보정 각속도를

$$
\omega_{posture,max}
=
\begin{bmatrix}
\omega_{x,posture,max}\\
\omega_{y,posture,max}\\
\omega_{z,posture,max}
\end{bmatrix}
$$

라 하면

$$
\boxed{
\omega_{posture}
=
\operatorname{sat}
(
\omega_{posture}^{raw},
-\omega_{posture,max},
\omega_{posture,max}
)
}
$$

로 제한한다.

단일 자세 PI가 직접 몸체 회전각을 출력하는 것은 아니다. 출력 $\omega_{posture}$는 몸체 자세 오버레이에 사용할 보정 각속도이다. 별도의 내부 각속도 PI는 두지 않고, Saturation과 Rate Limiter를 거친 값을 200 Hz로 적분한다.

$$
\eta_{posture,cmd}[k+1]
=
\operatorname{clamp}
\left(
\eta_{posture,cmd}[k]
+
\omega_{posture}[k]T_s,
-\eta_{posture,max},
\eta_{posture,max}
\right)
$$

자세 명령으로 몸체 기준 좌표계 $\{R\}$에서 실제 몸체 좌표계 $\{B\}$로의 회전행렬을 만든다.

$$
{}^RR_B
=
R_z(\psi_{posture,cmd})
R_y(\theta_{posture,cmd})
R_x(\phi_{posture,cmd})
$$

`Tripod_Enable`에 따라 기준 좌표계 발끝 목표를 먼저 선택한다. `Tripod_Enable=1`이면 Tripod와 Contact Adaptation이 만든 Stance/Swing 목표를 사용하고, `Tripod_Enable=0`이면 6개 다리의 기본 또는 직전 안전 STANCE 목표를 계속 출력한다. 선택된 기준 발끝 목표를 IK 직전에 다음과 같이 역회전한다.

$$
\boxed{
{}^Bp_{F_i}^{cmd}
=
({}^RR_B)^T
{}^Rp_{F_i}^{base}
}
$$

$\{R\}$과 $\{B\}$는 같은 몸체 원점을 사용하므로 평행이동 없이 원점 기준 회전만 적용된다. 자세 오버레이는 Final Gait Body Twist, Tripod Phase, 보폭, Swing 착지점 계산에 입력하지 않는다. 따라서 보행 중에도 명목 Stance/Swing 궤적은 유지되고 몸체 자세만 변경된다.

조종 모드에서는 Roll·Pitch 자세 PI만 사용하고, 보정 모드에서는 Yaw 자세 PI만 사용한다. `Body_Control_Enable=1`, `Tripod_Enable=0`이면 6개 발을 모두 STANCE로 유지하여 보행 없이 몸체 자세만 보정한다. 암 모드와 LANDED, STANDING, LANDING, FAULT, KILL 상태에서는 자세 PI 출력과 자세 오버레이를 0으로 만들고 적분항을 누적하지 않는다. PI가 다시 활성화될 때는 해당 적분항과 자세 명령 상태를 초기화한다.

---

# 11. Final Gait Body Twist

조종 모드에서는 사용자 Feedforward 명령과 x·y Position PI 및 Gait Heading Yaw PI Feedback을 합한다. Roll·Pitch 자세 PI와 보정 모드 Yaw 자세 PI는 이 Body Twist에 합하지 않는다.

선속도는

$$
\boxed{
v_{gait}
=
v_{user}
+
v_{feedback}
}
$$

각속도는

$$
\boxed{
\omega_{gait}
=
\omega_{user}
+
\omega_{heading,feedback}
}
$$

이다.

따라서

$$
\boxed{
\xi_{gait}
=
\begin{bmatrix}
v_{gait}\\
\omega_{gait}
\end{bmatrix}
}
$$

가 된다.

즉,

$$
\xi_{gait}
=
\xi_{user}
+
\xi_{position}
+
\xi_{heading}
$$

의 구조이다.

조종 모드에서 현재 사용하는 성분을 직접 쓰면

$$
v_{gait}^{B}
=
\begin{bmatrix}
v_{x,user}\\0\\0
\end{bmatrix}
+
{}^BR_W
\begin{bmatrix}
v_{x,feedback}^{W}\\
v_{y,feedback}^{W}\\
0
\end{bmatrix}
$$

이고

$$
\omega_{gait}^{B}
=
\begin{bmatrix}
0\\
0\\
\omega_{z,user}+\omega_{z,heading,feedback}
\end{bmatrix}
$$

이다.

보정 모드의 x·y·z 이동속도는 PI 없이 발 고정 몸체 이동 경로에 직접 사용한다. 보정 모드 Yaw 자세 PI는 이 Body Twist에 합하지 않고 IK 직전 자세 오버레이에만 사용한다.

$$
v_{gait}^{B}
=
v_{correction}^{B}
$$

$$
\omega_{gait}^{B}
=
\begin{bmatrix}
0\\0\\0
\end{bmatrix}
$$

암 모드에는 Position PI와 자세 PI를 적용하지 않고 암 제어 명령만 해당 기능 블록으로 전달한다. LANDED, STANDING, LANDING, FAULT, KILL 상태에서는 Final Gait Body Twist를 0으로 만든다.

---

## 11.1 최종 Saturation

Final Gait Body Twist에도 물리적인 제한을 둔다.

$$
v_{gait,j}
=
\operatorname{clamp}
(
v_{gait,j},
-v_{j,max},
v_{j,max}
)
$$

$$
\omega_{gait,j}
=
\operatorname{clamp}
(
\omega_{gait,j},
-\omega_{j,max},
\omega_{j,max}
)
$$

---

## 11.2 Twist Rate Limiter

명령이 한 주기 만에 급변하지 않도록

$$
\Delta v_{max}
=
a_{B,max}T_s
$$

$$
\Delta\omega_{max}
=
\alpha_{B,max}T_s
$$

를 정의한다.

$$
v_{cmd}[k]
=
v_{cmd}[k-1]
+
\operatorname{clamp}
\left(
v_{gait}[k]-v_{cmd}[k-1],
-\Delta v_{max},
\Delta v_{max}
\right)
$$

$$
\omega_{cmd}[k]
=
\omega_{cmd}[k-1]
+
\operatorname{clamp}
\left(
\omega_{gait}[k]-\omega_{cmd}[k-1],
-\Delta\omega_{max},
\Delta\omega_{max}
\right)
$$

로 실제 Gait Generator에 전달할 Twist를 생성한다.

이후 Gait 식에서 $v_B$, $\omega_B$는 이 최종 제한을 통과한 보행 명령을 의미한다. 현재 $\omega_B$에는 조종 모드의 사용자 Yaw 각속도와 Gait Heading Yaw PI 보정만 포함하며 Roll·Pitch 및 보정 모드 Yaw 자세 PI는 포함하지 않는다.

---

# 12. Tripod Gait Manager

`Body_Control_Enable`은 Final Gait Body Twist와 Stance 궤적 갱신을 허용하는 상위 Enable이다. `Tripod_Enable`은 Tripod 위상 진행과 Swing 궤적을 허용하는 보행 Enable이다.

| Body_Control_Enable | Tripod_Enable | 동작 |
|---:|---:|---|
| 0 | 0 | 보행용 Body Twist와 자세 오버레이를 정지하고 상위 안전·서기·착지 블록의 발끝 목표를 사용한다. |
| 1 | 0 | 6개 다리의 기본 또는 직전 안전 STANCE 목표를 계속 출력하고 몸체 위치·자세 또는 보정 모드 명령을 수행한다. |
| 1 | 1 | Tripod 위상에 따라 STANCE와 SWING을 반복한다. |

`Body_Control_Enable=0`, `Tripod_Enable=1` 조합은 사용하지 않는다. `Tripod_Enable=0`은 발끝 목표 0을 의미하지 않으며, Tripod 위상만 정지하고 6개 STANCE 기준 목표를 유지한다. 조종 모드에서는 `Motion_Armed=1`일 때 `Body_Control_Enable=1`로 두고, 전진·후진 또는 Yaw 회전 명령이 임계값을 넘을 때만 `Tripod_Enable=1`로 둔다. 보정 모드에서는 `Body_Control_Enable=1`, `Tripod_Enable=0`으로 둔다.

Tripod 그룹은

$$
A=\{1,3,5\}
$$

$$
B=\{2,4,6\}
$$

로 정의한다.

Phase A:

$$
1,3,5:\ SWING
$$

$$
2,4,6:\ STANCE
$$

Phase B:

$$
1,3,5:\ STANCE
$$

$$
2,4,6:\ SWING
$$

이다.

---

## 12.1 Phase 진행률

한 Tripod Phase의 목표 시간을 $T_{phase}$라 하고 Phase 시작 시간을 $t_0$라 하면

$$
\tau
=
\frac{t-t_0}{T_{phase}}
$$

이다.

정규화 진행률은

$$
s_{phase}
=
\operatorname{clamp}
(
\tau,
0,
1
)
$$

로 둔다.

정상적인 경우 $s_{phase}=1$이 되면 다음 Phase로 전환한다.

단, Late Landing 상태에서는 다음 Phase 전환을 정지한다.

---

# 13. Stance Leg Trajectory

몸체 기준 좌표계 $\{R\}$에서 Stance 발끝 위치를

$$
p_i=
\begin{bmatrix}
x_i\\y_i\\z_i
\end{bmatrix}
$$

라 한다. 13장부터 Contact Adaptation까지의 $p_i$는 별도 표기가 없으면 기준 좌표계 $\{R\}$에서 표현한 명목 보행 발끝 위치를 의미한다.

지면 기준으로 발끝이 정지해 있다면 몸체 기준 좌표계에서 관측되는 발끝 속도는 강체 운동학에 의해

$$
\boxed{
\dot p_i
=
-v_B
-
\omega_B\times p_i
}
$$

이다.

Skew Matrix를 사용하면

$$
\dot p_i
=
-v_B
-
[\omega_B]_\times p_i
$$

로 표현된다.

---

## 13.1 200 Hz 이산화

Forward Euler를 사용하면

$$
\boxed{
p_i^{ref}[k+1]
=
p_i^{ref}[k]
+
\left(
-v_B[k]
-
\omega_B[k]\times p_i^{ref}[k]
\right)T_s
}
$$

이다.

$$
T_s=0.005\text{ s}
$$

이다.

이 궤적이 Stance 다리가 지면을 밀거나 당겨 몸체를 이동시키는 **PULL 궤적**이다.

---

## 13.2 Stance Workspace 제한

발끝이 작업공간 끝까지 끌려가지 않도록 각 Stance 발끝에 허용 영역을 둔다.

기준 발끝 위치를

$$
p_{0,i}
$$

라 하고 최대 허용 변위를

$$
\Delta p_{stance,max}
$$

라 하면

$$
\left\|
p_i^{ref}-p_{0,i}
\right\|
\le
\Delta p_{stance,max}
$$

가 되도록 제한한다.

작업공간 한계에 접근하면 다음 Swing Phase로 전환하거나 Body Twist를 감소시킨다.

---

# 14. Swing Landing Target

몸체 기준 좌표계 $\{R\}$에서 Swing 다리의 기준 발끝 위치를

$$
p_{0,i}
$$

라 한다.

다음 Stance 구간에서 몸체가 이동할 양의 절반을 선행 보상하여 착지 목표점을 생성한다.

$$
\boxed{
p_{r,i}
=
p_{0,i}
+
\frac{T_{stance}}{2}
\left(
v_B
+
\omega_B\times p_{0,i}
\right)
}
$$

여기서

$$
T_{stance}
$$

는 예상 Stance 시간이다.

---

## 14.1 선속도 항

$$
\Delta p_{v,i}
=
\frac{T_{stance}}{2}v_B
$$

는 몸체의 예상 병진 운동을 보상한다.

---

## 14.2 회전 항

$$
\Delta p_{\omega,i}
=
\frac{T_{stance}}{2}
\left(
\omega_B\times p_{0,i}
\right)
$$

는 몸체 회전에 의해 각 발끝이 이동해야 하는 위치를 보상한다.

---

## 14.3 착지점 제한

계산된 목표점은 각 다리의 안전 작업공간으로 제한한다.

$$
p_{r,i}
=
\operatorname{ProjectWorkspace}_i
(
p_{r,i}
)
$$

이 함수는 최소한

- IK 해 존재 여부
- 관절 제한
- 몸체와의 충돌 여부
- 최대 Step Length
- 방사 방향 오프셋이 적용된 전체 Swing 중간 궤적

를 검사해야 한다.

---

# 15. Swing Bezier and Radial Offset Trajectory

Swing 시작 위치를

$$
P_0=
\begin{bmatrix}
x_0\\y_0\\z_0
\end{bmatrix}
$$

착지 목표 위치를

$$
P_3=
\begin{bmatrix}
x_r\\y_r\\z_r
\end{bmatrix}
$$

라 한다.

Swing Height를 $h$라 하면

$$
P_1
=
P_0
+
\begin{bmatrix}
0\\0\\\frac{4}{3}h
\end{bmatrix}
$$

$$
P_2
=
P_3
+
\begin{bmatrix}
0\\0\\\frac{4}{3}h
\end{bmatrix}
$$

로 둔다.

3차 Bezier Curve는

$$
\boxed{
B(s)
=
(1-s)^3P_0
+
3(1-s)^2sP_1
+
3(1-s)s^2P_2
+
s^3P_3
}
$$

이다.

$$
0\le s\le1
$$

---

## 15.1 시간 진행률

Swing 궤적의 시간 진행에는 **Quintic Time Scaling을 사용한다.**

Swing 시작 시각을 $t_s$, Swing 시간을 $T_{swing}$이라 하면 먼저 현재 Swing 진행률을

$$
\tau
=
\operatorname{clamp}
\left(
\frac{t-t_s}{T_{swing}},
0,
1
\right)
$$

로 계산한다.

여기서

$$
0\le\tau\le1
$$

이며,

- $\tau=0$: Swing 시작
- $\tau=1$: Swing 종료

를 의미한다.

Bezier Curve에 직접 $\tau$를 넣지 않고, 다음 Quintic Time Scaling을 적용하여 실제 궤적 진행 변수 $s$를 계산한다.

$$
\boxed{
s(\tau)
=
10\tau^3
-
15\tau^4
+
6\tau^5
}
$$

이 함수는 Swing 시작과 종료에서

$$
\dot s(0)=0,
\qquad
\dot s(1)=0
$$

및

$$
\ddot s(0)=0,
\qquad
\ddot s(1)=0
$$

을 만족한다.

따라서 발끝이 Swing 시작 순간에 갑자기 움직이거나 착지 순간에 갑자기 정지하는 현상을 줄이고, 시작과 종료를 부드럽게 만든다.

## 15.2 Swing Radial Offset

i번 다리의 Body 좌표계 기준 장착 방향을 $\alpha_i$라 하고, 다리 바깥 방향 단위벡터를

$$
\boxed{
e_{r,i}
=
\begin{bmatrix}
\cos\alpha_i\\
\sin\alpha_i\\
0
\end{bmatrix}
}
$$

로 정의한다. 각 다리의 장착 방향은

$$
\alpha_i
=
\left[
-45^\circ,
-90^\circ,
-135^\circ,
45^\circ,
90^\circ,
135^\circ
\right]
$$

이다.

Swing 최고점의 방사 방향 오프셋을 $r_{swing}$이라 하면 진행률에 따른 오프셋은

$$
\boxed{
r(s)
=
4r_{swing}s(1-s)
}
$$

로 계산한다. $r(0)=r(1)=0$이고 $r(0.5)=r_{swing}$이므로 시작점과 착지점은 바뀌지 않고 Swing 최고점에서만 최대 오프셋이 적용된다.

최종 Swing 발끝 목표 위치는

$$
\boxed{
p_i^{ref}
=
B\left(s(\tau)\right)
+
r\left(s(\tau)\right)e_{r,i}
}
$$

로 계산한다. 현재 Simulink 구현에서는 모든 Swing 다리에

$$
\boxed{
r_{swing}=0.07\text{ m}
}
$$

를 동일하게 적용한다.

즉, 매 제어 주기에서

```text
현재 시간
   ↓
Swing 진행률 τ 계산
   ↓
Quintic Time Scaling
s = 10τ³ - 15τ⁴ + 6τ⁵
   ↓
Bezier Curve B(s)
   ↓
방사 방향 오프셋 r(s)e_r,i 추가
   ↓
발끝 목표 위치 p_i^ref
```

순서로 Swing 궤적을 생성한다.


## 15.3 Swing Height

Swing 다리의 기본 발끝 상승 높이를 $h_0$로 정의한다.

현재 Simulink 검증에서는

$$
\boxed{
h_0=0.25\text{ m}
}
$$

를 사용한다.

몸체 기준점 $O_R$에 대한 현재 몸체 원점 $O_B$의 z방향 상대 변위를

$$
\boxed{
\Delta z_B
=
z_{O_B}-z_{O_R}
}
$$

로 정의한다.

따라서

- $\Delta z_B=0$: 몸체 원점이 기준 높이에 있음
- $\Delta z_B>0$: 몸체 원점이 기준점보다 위에 있음
- $\Delta z_B<0$: 몸체 원점이 기준점보다 아래에 있음

을 의미한다.

Swing Height는 몸체의 상대 높이에 따라 다음과 같이 보정한다.

$$
\boxed{
h
=
\operatorname{clamp}
\left(
h_0+k_h\Delta z_B,
h_{\min},
h_{\max}
\right)
}
$$

여기서

- $h_0$: 기준 Swing Height
- $k_h$: 몸체 높이에 따른 Swing Height 보정 Gain
- $h_{\min}$: 최소 Swing Height
- $h_{\max}$: 최대 Swing Height

이다.

즉, 몸체 원점이 기준점보다 높아지면 Swing Height를 증가시키고, 몸체 원점이 기준점보다 낮아지면 Swing Height를 감소시킨다.

현재 Position PI는 x·y 위치에만 적용하므로 z축 위치 오차와 z축 Position PI는 사용하지 않는다. $\Delta z_B$는 Position PI와 독립적인 Swing Height 보정값으로만 사용한다. 보정 모드의 z축 이동속도 명령에도 Position PI를 적용하지 않는다.

최종적으로 Swing Bezier Curve의 제어점은 계산된 $h$를 이용하여

$$
P_1
=
P_0
+
\begin{bmatrix}
0\\
0\\
\frac{4}{3}h
\end{bmatrix}
$$

$$
P_2
=
P_3
+
\begin{bmatrix}
0\\
0\\
\frac{4}{3}h
\end{bmatrix}
$$

로 생성한다.


## 15.4 현재 Simulink 검증값

현재 구현의 보행 및 초기자세 파라미터는 다음과 같다.

| 항목 | 값 |
|---|---:|
| 제어 주기 | `0.005 s` |
| Tripod Phase 시간 | `0.5 s` |
| 전진 시험 속도 | `0.14 m/s` |
| 한 Phase 발 이동량 | `0.07 m` |
| Swing Height | `0.25 m` |
| Swing Radial Offset | `0.07 m` |
| 초기 관절각 | $q_1=0^\circ$, $q_2=30^\circ$, $q_3=50^\circ$ |

6개 다리의 전체 Swing 및 Stance 궤적을 검사한 결과는 다음과 같다.

| 항목 | 범위 또는 결과 |
|---|---:|
| $q_1$ | $-17.75^\circ \sim 16.30^\circ$ |
| $q_2$ | $-79.79^\circ \sim 47.08^\circ$ |
| $q_3$ | $12.53^\circ \sim 125.53^\circ$ |
| 전체 최대 관절각 | $125.53^\circ$ |
| $\pm135^\circ$ 제한 여유 | $9.47^\circ$ |
| IK 작업공간 초과 | 없음 |


# 16. 발 접촉 판단

i번 발의 압력센서 측정값을

$$
F_i[k]
$$

라 한다.

Contact Threshold와 Release Threshold를 각각

$$
F_{contact}
$$

$$
F_{release}
$$

로 둔다.

반드시

$$
\boxed{
F_{release}
<
F_{contact}
}
$$

가 되도록 하여 Hysteresis를 만든다.

---

## 16.1 Contact 판정

연속 $N_c$ sample 동안

$$
F_i>F_{contact}
$$

이면

$$
Contact_i=1
$$

로 변경한다.

200 Hz 제어에서 접촉 확인 시간은

$$
T_{contact}
=
N_cT_s
=
N_c\times0.005\text{ s}
$$

이다.

원하는 실제 접촉 확인 시간 $T_{contact}$에 맞춰

$$
N_c
=
\frac{T_{contact}}{0.005}
$$

로 Sample 수를 정한다.

---

## 16.2 Release 판정

연속 $N_r$ sample 동안

$$
F_i<F_{release}
$$

이면

$$
Contact_i=0
$$

으로 변경한다.

Release 확인 시간은

$$
T_{release}
=
N_rT_s
=
N_r\times0.005\text{ s}
$$

이다.

원하는 실제 Release 확인 시간 $T_{release}$에 맞춰

$$
N_r
=
\frac{T_{release}}{0.005}
$$

로 Sample 수를 정한다.

---

# 17. Early Landing

Swing이 정상적으로 종료되기 전에 발 접촉이 감지되면 Early Landing으로 판단한다.

노이즈 또는 Swing 초기에 발생할 수 있는 오검출을 방지하기 위해 Early Landing은 Swing 후반의 하강 구간에서만 판단한다.

예를 들어

$$
s>s_{early,min}
$$

이고

$$
\dot z_i<0
$$

이며

$$
Contact_i=1
$$

이면 Early Landing이 발생한 것으로 판단한다.

Early Landing은 별도의 지속 상태로 유지하지 않고, **접촉이 검출된 다리를 즉시 STANCE 상태로 전환한다.**

접촉이 검출된 제어 주기를 $k_{contact}$라 하면 해당 순간의 발끝 위치를

$$
\boxed{
p_{contact,i}
=
p_i^{ref}[k_{contact}]
}
$$

로 저장한다.

그리고 같은 제어 주기에서는 발끝 목표 위치를

$$
p_i^{ref}[k_{contact}]
=
p_{contact,i}
$$

로 유지하여 Bezier 하강 궤적을 즉시 중단한다.

동시에 Stance 궤적의 시작 위치를

$$
\boxed{
p_{stance,i}[k_{contact}]
=
p_{contact,i}
}
$$

로 초기화하고

$$
\boxed{
State_i=STANCE
}
$$

로 전환한다.

다음 200 Hz 제어 주기부터 해당 다리는 일반적인 Stance 궤적

$$
\dot p_i
=
-v_B-\omega_B\times p_i
$$

을 적용하며,

$$
p_i^{ref}[k+1]
=
p_i^{ref}[k]
+
\left(
-v_B[k]
-
\omega_B[k]\times p_i^{ref}[k]
\right)T_s
$$

로 발끝 목표 위치를 갱신한다.

따라서 Early Landing 처리 흐름은 다음과 같다.

```text
SWING
  ↓
Swing 후반 하강 중 CONTACT 검출
  ↓
현재 발끝 위치 저장
  ↓
Bezier 하강 즉시 중단
  ↓
State_i = STANCE
  ↓
다음 200 Hz 주기부터 Stance Trajectory
```

Early Landing이 발생하지 않은 다른 Swing 다리는 기존 Bezier 궤적을 계속 수행한다.

예를 들어 1, 3, 5번 다리가 Swing 중이고 1번 다리만 먼저 접촉했다면

```text
1 : STANCE
3 : SWING
5 : SWING
```

으로 관리한다.

다음 Tripod Phase는 Swing 그룹에 속했던 모든 다리가 지면 접촉을 완료한 이후에만 시작한다.

---


# 18. Late Landing

정상 Swing 시간이 종료되어

$$
s=1
$$

이 되었지만

$$
Contact_i=0
$$

이면

$$
\boxed{
State_i=LATE\_LANDING
}
$$

으로 전환한다.

Late Landing 중에는 다음 Tripod Phase 전환을 정지한다.

---

## 18.1 Search Down

현재 몸체 기준 좌표계의 $-z_R$ 방향으로 발끝을 내리는 동시에 각 다리 장착점 방향으로 당겨 지면을 탐색한다. 하강만 수행할 때 발생하는 다리 과신전과 IK 작업공간 부담을 줄이기 위한 동작이다.

다리 $i$의 바깥쪽 방사 단위벡터를

$$
e_{r,i}
=
\begin{bmatrix}
\cos\alpha_i\\
\sin\alpha_i\\
0
\end{bmatrix}
$$

로 정의한다. 안쪽 이동 속도는

$$
\boxed{
v_{in}
=
k_{in}v_{search}
}
$$

로 계산한다. 현재 Simulink 검증값은

$$
v_{search}=0.20\text{ m/s},
\qquad
k_{in}=0.8,
\qquad
v_{in}=0.16\text{ m/s}
$$

이다.

$$
{}^B\dot p_{late}
=
\begin{bmatrix}
-v_{in}\cos(\alpha_i)\\
-v_{in}\sin(\alpha_i)\\
-v_{search}
\end{bmatrix}
$$

따라서

$$
x_i[k+1]
=
x_i[k]
-
v_{in}\cos(\alpha_i)T_s
$$

$$
y_i[k+1]
=
y_i[k]
-
v_{in}\sin(\alpha_i)T_s
$$

$$
\boxed{
z_i[k+1]
=
z_i[k]
-
v_{search}T_s
}
$$

이다.

---

## 18.2 Late Landing 종료

탐색 중

$$
Contact_i=1
$$

이 되면

$$
State_i=STANCE
$$

로 전환한다.

Swing 그룹의 모든 다리가 접촉하면 다음 Tripod Phase로 전환한다.

---

## 18.3 Late Landing Fault

검색 시작 위치를 $p_{late,start,i}$라 하면 전체 대각 탐색 거리는

$$
d_{search}
=
\left\|
p_i-p_{late,start,i}
\right\|_2
$$

이다. 필요하면 수직 하강 거리 $|z_i-z_{late,start,i}|$와 안쪽 이동 거리를 별도로 제한할 수 있다.

다음 중 하나라도 만족하면 Fault로 전환한다.

$$
d_{search}>d_{search,max}
$$

또는

$$
t_{search}>T_{search,max}
$$

또는

$$
IK\_Valid=0
$$

또는

$$
JointLimit=1
$$

즉,

$$
\boxed{
LateLandingFault
=
D\lor T\lor IK\lor J
}
$$

로 볼 수 있다.

Recovery 동작은 별도 설계 대상으로 둔다.

---

# 19. 다리별 최종 발끝 목표 위치

각 다리의 현재 상태에 따라 최종 발끝 목표 위치를 선택한다.

$$
p_i^{ref}
=
\begin{cases}
p_{stance,i},
&
State_i=STANCE
\\[4pt]
p_{bezier,i},
&
State_i=SWING
\\[4pt]
p_{late,i},
&
State_i=LATE\_LANDING
\end{cases}
$$

Early Landing은 별도의 지속 상태로 두지 않는다. Swing 중 Early Landing이 검출되면 해당 제어 주기에서 현재 발끝 위치를 Stance 시작 위치로 저장한 뒤 즉시

$$
State_i=STANCE
$$

로 전환한다.

따라서 다음 제어 주기부터는 위 식의 `STANCE` 항을 사용한다.

매 200 Hz 주기마다

$$
p_1^{ref},
p_2^{ref},
\dots,
p_6^{ref}
$$

의 6개 발끝 목표 위치가 생성된다.

---


# 20. 몸체 자세 오버레이 및 다리 좌표계 변환

## 20.1 몸체 원점 기준 자세 오버레이

자세 오버레이 앞의 Foot Target Selector는 몸체 기준 좌표계 $\{R\}$의 기준 발끝 목표 ${}^Rp_{F_i}^{base}$를 선택한다.

$$
{}^Rp_{F_i}^{base}
=
\begin{cases}
{}^Rp_{F_i}^{tripod}, & Tripod\_Enable=1\\
{}^Rp_{F_i}^{stance\_hold}, & Tripod\_Enable=0
\end{cases}
$$

$p_{F_i}^{stance\_hold}$는 기본 자세 발끝 위치 또는 직전 안전 STANCE 위치이다. `Tripod_Enable=0`이어도 이 목표를 매 주기 출력하므로 정지 상태의 Roll·Pitch 또는 보정 Yaw 자세 명령도 정상적으로 적용된다. 조종 모드 Roll·Pitch와 보정 모드 Yaw 자세 PI는 선택 전의 Tripod 궤적을 직접 수정하지 않는다.

단일 자세 PI의 보정 각속도를 적분한 $\eta_{posture,cmd}$로

$$
{}^RR_B
=
R_z(\psi_{posture,cmd})
R_y(\theta_{posture,cmd})
R_x(\phi_{posture,cmd})
$$

를 만들고, IK에 넣을 몸체 좌표계 발끝 목표를

$$
\boxed{
{}^Bp_{F_i}^{cmd}
=
({}^RR_B)^T
{}^Rp_{F_i}^{base}
}
$$

로 계산한다. $O_R$과 $O_B$를 같은 회전 중심으로 사용하므로 평행이동 없이 몸체 원점 기준 Roll·Pitch·Yaw 회전만 적용된다. 자세 오버레이는 Stance/Swing 상태, 보폭, 진행률, Swing 착지점을 변경하지 않는다.

## 20.2 몸체 좌표계 → 다리 좌표계 변환

자세 PI에서는 Euler Angle 오차를 사용하지만, **서로 다른 좌표계 사이의 벡터와 점을 변환할 때는 회전행렬을 사용한다.**

IMU에서 얻은 Roll, Pitch, Yaw를

$$
\phi,\qquad \theta,\qquad \psi
$$

라 하면 Z-Y-X 순서의 몸체 회전행렬은

$$
{}^WR_B
=
R_z(\psi)R_y(\theta)R_x(\phi)
$$

로 생성하여 절대좌표계의 상태 추정과 벡터 변환에 사용한다. 이 실제 자세행렬 ${}^WR_B$는 몸체 자세 오버레이용 상대 회전행렬 ${}^RR_B$와 구분한다.


i번 다리 좌표계 원점의 몸체 좌표계 위치를

$$
{}^Bp_{L_i}
$$

라 한다.

다리 좌표계에서 몸체 좌표계로의 회전행렬을

$$
{}^BR_{L_i}
$$

라 하면

$$
{}^Bp_F
=
{}^Bp_{L_i}
+
{}^BR_{L_i}
{}^{L_i}p_F
$$

이다.

따라서 자세 오버레이가 적용된 몸체 좌표계 발끝 목표 위치를 다리 좌표계로 변환하면

$$
\boxed{
{}^{L_i}p_F^{ref}
=
({}^BR_{L_i})^T
\left(
{}^Bp_F^{cmd}
-
{}^Bp_{L_i}
\right)
}
$$

이다.

이 값을 각 다리의 IK 입력으로 사용한다.

---

# 21. 3DOF Inverse Kinematics

> 아래 식은 일반적인 `Yaw - Pitch - Pitch` 형식의 3DOF 다리를 기준으로 한 기본식이다. 실제 로봇의 각 관절 0도 기준, 회전 부호, 서보 조립 방향은 좌표축 README와 실제 기구에 맞추어 `sign`과 `offset`을 적용해야 한다.

다리 링크 길이를

$$
L_1,\qquad
L_2,\qquad
L_3
$$

라 한다.

다리 좌표계에서 목표 발끝 위치를

$$
{}^{L_i}p_F^{ref}
=
\begin{bmatrix}
x\\y\\z
\end{bmatrix}
$$

라 한다.

---

## 21.1 1번 관절

첫 번째 관절이 z축 주위 Yaw 회전이라고 하면

$$
\boxed{
\theta_1
=
\operatorname{atan2}(y,x)
}
$$

이다.

수평 반경은

$$
r
=
\sqrt{x^2+y^2}
$$

이고 첫 번째 링크 길이를 제외한 평면 거리 성분은

$$
\rho=r-L_1
$$

이다.

---

## 21.2 2-3번 관절 평면 문제

2번 관절에서 발끝까지의 거리는

$$
d
=
\sqrt{\rho^2+z^2}
$$

이다.

Cosine Law를 이용하면

$$
c_3
=
\frac{
\rho^2+z^2-L_2^2-L_3^2
}{
2L_2L_3
}
$$

이다.

IK 해가 존재하려면

$$
\boxed{
-1\le c_3\le1
}
$$

이어야 한다.

3번 관절은

$$
\boxed{
\theta_3
=
\operatorname{atan2}
\left(
\sigma_3\sqrt{1-c_3^2},
c_3
\right)
}
$$

로 계산한다.

여기서

$$
\sigma_3\in\{-1,+1\}
$$

은 기구의 Knee-Up / Knee-Down 해 선택에 사용한다.

2번 관절은

$$
\boxed{
\theta_2
=
\operatorname{atan2}(z,\rho)
-
\operatorname{atan2}
\left(
L_3\sin\theta_3,
L_2+L_3\cos\theta_3
\right)
}
$$

로 계산한다.

---

## 21.3 실제 서보 각도 변환

기구학 각도와 실제 서보 방향이 다를 수 있으므로

$$
\boxed{
\theta_{servo,ij}
=
s_{ij}\theta_{ij}
+
\theta_{offset,ij}
}
$$

를 사용한다.

여기서

$$
s_{ij}\in\{-1,+1\}
$$

은 서보 회전 방향 보정값이며

$$
\theta_{offset,ij}
$$

는 조립 시 Zero Offset이다.

---

# 22. IK Workspace 검사

평면 2-Link 부분은 최소한

$$
|L_2-L_3|
\le
d
\le
L_2+L_3
$$

를 만족해야 한다.

또한

$$
-1\le c_3\le1
$$

이어야 한다.

따라서

$$
IK\_Valid
=
W_d
\land
W_c
\land
W_{joint}
$$

로 검사한다.

해가 존재하지 않으면 새로운 관절 명령을 Servo에 전달하지 않는다.

---

# 23. 관절 각도 제한

모든 관절의 허용 범위는

$$
\boxed{
-135^\circ
\le
\theta_{ij}
\le
135^\circ
}
$$

이다.

따라서

$$
\theta_{ij}^{lim}
=
\operatorname{clamp}
(
\theta_{ij}^{ref},
-135^\circ,
135^\circ
)
$$

를 적용한다.

가능하면 단순 clamp 전에 $IK_{Valid}=0$으로 판단하여 비정상적인 발끝 목표점 자체를 수정하는 것이 좋다.

---

# 24. 관절 최대 각속도 제한

서보의 최대 허용 각속도를

$$
\dot\theta_{max}
$$

라 하면 한 제어 주기에서 허용되는 최대 관절각 변화량은

$$
\Delta\theta_{max}
=
\dot\theta_{max}T_s
$$

이다.

따라서

$$
\boxed{
\theta_{ij}^{cmd}[k]
=
\theta_{ij}^{cmd}[k-1]
+
\operatorname{clamp}
\left(
\theta_{ij}^{lim}[k]
-
\theta_{ij}^{cmd}[k-1],
-\Delta\theta_{max},
\Delta\theta_{max}
\right)
}
$$

로 제한한다.

---

# 25. 비정상 목표각 Jump 검사

Rate Limiter와 별도로 IK 오류 또는 좌표변환 오류에 의해 관절 목표각이 비정상적으로 급변하는 경우를 검출하기 위해 Jump Threshold를 둔다.

현재 IK 목표각과 직전 Servo 명령각의 차이를

$$
\Delta\theta_{ij}^{raw}
=
\left|
\theta_{ij}^{ref}[k]
-
\theta_{ij}^{cmd}[k-1]
\right|
$$

로 계산한다.

만약

$$
\boxed{
\Delta\theta_{ij}^{raw}
>
\Delta\theta_{jump,max}
}
$$

이면 해당 관절의 새 목표각을 비정상 명령으로 판단한다.

비정상 Jump가 검출되면 새 목표각을 Servo에 전달하지 않고 **직전 명령각을 그대로 HOLD한다.**

$$
\boxed{
\theta_{ij}^{cmd}[k]
=
\theta_{ij}^{cmd}[k-1]
}
$$

동시에 Joint Jump Fault Flag를 기록한다.

```text
IK 목표각
   ↓
Jump Threshold 검사
   ├─ 정상 → Rate Limiter → Servo Output
   └─ 비정상 → 직전 관절각 HOLD + Fault Flag
```

---


# 26. 지지다각형 검사

현재 지면에 접촉한 Stance 다리의 발끝 지면 투영점을 이용하여 지지다각형을 생성한다.

Tripod Gait에서는 세 개의 Stance 발의 지면 투영점

$$
P_1,\qquad P_2,\qquad P_3
$$

이 지지 삼각형을 구성한다.

몸체 원점의 지면 투영점을

$$
P_B^{xy}
$$

라 한다.

## 26.1 안전 지지다각형

실제 지지다각형 경계까지 정상 영역으로 사용하지 않고, **지지다각형을 중심 기준으로 10% 축소한 영역을 안전 지지다각형으로 사용한다.**

먼저 세 Stance 발의 중심점을

$$
\boxed{
C
=
\frac{P_1+P_2+P_3}{3}
}
$$

로 계산한다.

각 지지점은 중심 $C$를 기준으로 선형 크기의 90% 위치로 이동시킨다.

$$
\boxed{
P_i^{safe}
=
C
+
0.9
\left(
P_i-C
\right)
}
$$

따라서 $P_1^{safe},P_2^{safe},P_3^{safe}$가 Safe Support Polygon을 구성한다.

안전 지지다각형의 선형 Scale Factor는

$$
\boxed{
k_{support}=0.9
}
$$

이다.

## 26.2 안전 영역 내부 검사

안전 지지 삼각형의 세 꼭짓점을

$$
A=P_1^{safe},
\qquad
B=P_2^{safe},
\qquad
C=P_3^{safe}
$$

라 하고 몸체 원점의 지면 투영점을

$$
P=P_B^{xy}
$$

라 한다.

Barycentric Coordinate를 이용하여

$$
D
=
(y_B-y_C)(x_A-x_C)
+
(x_C-x_B)(y_A-y_C)
$$

$$
\lambda_A
=
\frac{
(y_B-y_C)(x_P-x_C)
+
(x_C-x_B)(y_P-y_C)
}{
D
}
$$

$$
\lambda_B
=
\frac{
(y_C-y_A)(x_P-x_C)
+
(x_A-x_C)(y_P-y_C)
}{
D
}
$$

$$
\lambda_C
=
1-\lambda_A-\lambda_B
$$

를 계산한다.

$$
\boxed{
\lambda_A\ge0,
\qquad
\lambda_B\ge0,
\qquad
\lambda_C\ge0
}
$$

이면 몸체 원점이 안전 지지다각형 내부에 있다고 판단하고 정상 보행을 계속한다.

반대로

$$
\boxed{
P_B^{xy}
\notin
\mathcal P_{safe}
}
$$

이면 몸체 원점이 안전 지지 영역을 벗어난 것으로 판단하고 **즉시 보행을 중단한다.**

별도의 Barycentric Margin은 사용하지 않고, 지지다각형 자체를 90%로 축소하여 안전 여유를 확보한다.

---


# 27. 전복 상태 검사

Roll과 Pitch가 정상 보행 제어 범위를 크게 벗어나면 안전장치에서 전복 상태로 처리한다.

전복 기준은

$$
\boxed{
|\phi|\ge80^\circ
\quad\lor\quad
|\theta|\ge80^\circ
}
$$

이다.

즉,

- $|Roll|\ge80^\circ$
- 또는 $|Pitch|\ge80^\circ$

중 하나라도 만족하면 **ROLLOVER FAULT**로 판단한다.

$$
RolloverFault
=
\left(
|\phi|\ge80^\circ
\right)
\lor
\left(
|\theta|\ge80^\circ
\right)
$$

ROLLOVER FAULT가 발생하면 해당 상태는 정상적인 Body Attitude PI 제어 범위를 벗어난 것으로 판단하고 **로봇을 즉시 정지한다.**

전복 상태에 대한 자동 Recovery는 수행하지 않는다.

---


# 28. Servo Output

모든 안전장치를 통과한 관절 목표각만 Servo Output으로 전달한다.

사용하는 Servo Motor는 **DS51150-270**이며 PWM 신호로 제어한다.

각 관절의 사용 각도 범위는

$$
\boxed{
-135^\circ
\le
\theta_{ij}^{cmd}
\le
135^\circ
}
$$

이다.

관절의 $0^\circ$ 기준 방향은 좌표축 README에서 정의한 각 관절의 기준 방향을 따른다.

## 28.1 PWM 설정

| 항목 | 값 |
|---|---:|
| PWM 주기 | 5 ms |
| PWM 주파수 | 200 Hz |
| 중립 Pulse | 1500 us |
| 사용 범위 | 500 ~ 2500 us |
| 관절 사용 각도 범위 | -135° ~ 135° |

Servo PWM은 Timer에서 200 Hz로 계속 출력하고, 전체 제어 루프도 200 Hz로 동작한다.

$$
\boxed{
T_{PWM}=T_s=5\text{ ms}
}
$$

따라서 **매 제어 주기마다 새로운 관절 목표각과 Pulse Width를 갱신**하며, 제어 루프 1회와 Servo PWM 주기 1회가 1:1로 대응한다.

## 28.2 관절각 → Pulse Width 변환

관절각

$$
-135^\circ
\le
\theta_{ij}^{cmd}
\le
135^\circ
$$

를

$$
500\ \mathrm{us}
\le
PWM_{ij}
\le
2500\ \mathrm{us}
$$

에 선형 대응시킨다.

전체 각도 범위는 $270^\circ$, Pulse Width 범위는 $2000\ \mathrm{us}$이므로

$$
\frac{2000}{270}
\approx
7.4074\ \mathrm{us/deg}
$$

이다.

따라서 최종 변환식은

$$
\boxed{
PWM_{ij}
=
1500
+
\frac{1000}{135}
\theta_{ij}^{cmd}
}
$$

이며 단위는 us이다.

$$
\theta_{ij}^{cmd}=-135^\circ
\Rightarrow
PWM_{ij}=500\ \mathrm{us}
$$

$$
\theta_{ij}^{cmd}=0^\circ
\Rightarrow
PWM_{ij}=1500\ \mathrm{us}
$$

$$
\theta_{ij}^{cmd}=135^\circ
\Rightarrow
PWM_{ij}=2500\ \mathrm{us}
$$

안전을 위해 최종 Pulse Width에는

$$
\boxed{
PWM_{ij}
=
\operatorname{clamp}
\left(
PWM_{ij},
500,
2500
\right)
}
$$

를 적용한다.

## 28.3 최종 Servo 명령 벡터

최종 관절 명령 벡터는

$$
q^{cmd}
=
\begin{bmatrix}
q_1^{cmd}\\
q_2^{cmd}\\
\vdots\\
q_6^{cmd}
\end{bmatrix}
$$

이고,

$$
q_i^{cmd}
=
\begin{bmatrix}
\theta_{i1}^{cmd}\\
\theta_{i2}^{cmd}\\
\theta_{i3}^{cmd}
\end{bmatrix}
$$

이다.

총 18개 관절의 목표각을 각각 PWM Pulse Width로 변환하여 Servo에 출력한다.

---


# 29. 제어 상태별 우선순위

## 29.1 상위 제어 상태 우선순위

조종기 명령과 로봇의 상위 제어 상태는 다음 우선순위를 따른다.

$$
\boxed{
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
READY(MANUAL/CORRECTION)
}
$$

상위 상태에서 허용되지 않은 기능의 Enable과 사용자 명령은 0으로 차단한다.

## 29.2 다리 궤적 상태 우선순위

발끝 명령과 다리별 안전 상태는 다음 우선순위를 따른다.

$$
\boxed{
ROLLOVER\_FAULT
>
FAULT
>
LATE\_LANDING
>
SWING/STANCE
}
$$

Early Landing은 별도의 지속 상태가 아니라 **SWING → STANCE 즉시 전환 이벤트**로 처리한다.

즉, Early Landing이 검출된 다리는 현재 Bezier 하강을 중단하고 즉시 STANCE로 전환하며, 다음 제어 주기부터 Stance Trajectory를 수행한다.

Late Landing 중에는 일반적인 Phase Time이 끝났더라도 다음 Tripod Phase로 넘어가지 않는다.

---


# 30. 200 Hz 전체 제어식

한 제어 주기 $k$에서 다음 순서로 계산한다.

## Step 1. Sensor Update

이번 제어 주기에 사용할 **각 다리의 현재 관절각, 발 압력센서값, 몸체 자세값을 읽고 좌표변환용 회전행렬을 계산한다.**

각 다리의 현재 관절각 벡터는

$$
q_i[k]
=
\begin{bmatrix}
\theta_{i1}[k]\\
\theta_{i2}[k]\\
\theta_{i3}[k]
\end{bmatrix}
$$

이다.

즉, $q_i[k]$는 $i$번 다리의 1번, 2번, 3번 관절의 현재 각도를 나타낸다.

각 발의 압력센서값은

$$
F_i[k]
$$

이며, 이후 Contact State에서 해당 발이 지면에 접촉했는지 판단하는 데 사용한다.

IMU에서는 현재 몸체의 자세각

$$
\phi[k],\qquad
\theta[k],\qquad
\psi[k]
$$

를 읽는다.

여기서

- $\phi[k]$: 현재 Roll
- $\theta[k]$: 현재 Pitch
- $\psi[k]$: 현재 Yaw

이다.

좌표변환에 사용하는 회전행렬 $R[k]$는 센서에서 직접 읽는 값이 아니라 현재 Roll, Pitch, Yaw를 이용해 계산한다.

$$
\boxed{
R[k]
=
R_z\left(\psi[k]\right)
R_y\left(\theta[k]\right)
R_x\left(\phi[k]\right)
}
$$

따라서 Step 1에서 직접 읽는 값은

$$
q_i[k],
\qquad
F_i[k],
\qquad
\phi[k],
\qquad
\theta[k],
\qquad
\psi[k]
$$

이고, 이 값을 이용해 $R[k]$를 계산한다.


## Step 2. RC Input Filter

조종기 입력의 작은 떨림과 노이즈를 제거하여 안정적인 사용자 명령값으로 만든다.

$$
u
\rightarrow
DeadZone
\rightarrow
LPF
$$

## Step 3. User Command

조종 모드에서는 필터링된 입력을 전진·후진 선속도, Roll·Pitch 목표각과 Yaw 각속도 명령으로 변환한다.

$$
v_{user}[k],
\quad
\phi_{ref}[k],
\quad
\theta_{ref}[k],
\quad
\omega_{z,user}[k]
$$

을 계산한다.

보정 모드에서는 x·y·z 이동속도와 Yaw 보정 목표각 $\psi_{corr,ref}$를 생성한다. 암 모드 입력은 몸체 Position PI와 자세 PI 경로로 보내지 않는다.

## Step 4. Attitude Reference Update

필터링된 Roll·Pitch 입력을 각 축의 목표각에 직접 대응시킨다.

$$
\phi_{ref}[k]
=
\phi_{\max}u_{R,f}[k]
$$

$$
\theta_{ref}[k]
=
\theta_{\max}u_{P,f}[k]
$$

조종 모드에서는 Yaw 각속도 명령을 적분하여 Heading Reference를 갱신한다.

$$
\psi_{ref}[k+1]
=
\operatorname{wrap}_{\pi}
\left(
\psi_{ref}[k]
+
\omega_{z,user}[k]T_s
\right)
$$

보정 모드의 Yaw 입력은 적분하지 않고 $\psi_{corr,ref}$에 직접 대응시킨다.

## Step 5. Forward Kinematics

현재 관절각을 이용해 각 다리 발끝의 현재 위치를 계산한다.

$$
{}^Bp_{F_i}^{meas}
=
FK_i(q_i)
$$

## Step 6. Contact State

각 발의 압력센서 값을 이용해 해당 발이 지면에 접촉했는지 판단한다.

$$
F_i
\rightarrow
Contact_i
$$

## Step 7. Body Position Estimate

STANCE이면서 실제 접촉 중인 다리들의 정보를 이용해 현재 몸체 원점 위치를 추정한다.

STANCE 상태이면서 CONTACT가 확인된 유효 다리 집합을 $\mathcal S$라 하고 $N_S=|\mathcal S|$라 하면

$$
{}^W\hat p_B
=
\frac{1}{N_S}
\sum_{i\in\mathcal S}
{}^W\hat p_{B,i}
$$

로 단순 평균한다.

## Step 8. Position Feedback

조종 모드에서 Throttle 선속도를 Yaw Reference 방향으로 적분하여 절대좌표계의 목표 x·y 위치를 생성한다.

$$
p_{B,ref}^{xy}[k]
=
p_{B,ref}^{xy}[k-1]
+
\begin{bmatrix}
\cos\psi_{ref}[k]\\
\sin\psi_{ref}[k]
\end{bmatrix}
v_{x,user}[k]T_s
$$

Position Reference Generator를 활성화할 때는 $p_{B,ref}^{xy}[k_0]={}^W\hat p_B^{xy}[k_0]$로 초기화한다.

목표 몸체 위치와 FK·Stance Anchor로 추정한 현재 Body Position의 차이를 PI 제어기에 넣어 위치 보정 속도를 계산한다.

$$
e_p^{xy}
=
p_{B,ref}^{xy}
-
{}^W\hat p_B^{xy}
$$

$$
v_{feedback}^{xy}
=
PI_p(e_p^{xy})
$$

$v_{x,user}$는 목표 위치 생성에 사용되는 동시에 Final Gait Body Twist의 Feedforward 선속도로 사용된다. $v_{feedback}^{xy}$는 목표 위치와 FK 기반 추정 위치 사이의 x·y 오차만 보정한다. 보정 모드의 x·y·z 이동속도와 암 모드에는 Position PI를 적용하지 않는다.

## Step 9. Gait Heading and Body Posture Feedback

조종 모드 Yaw는 적분된 Heading Reference와 IMU Yaw를 비교하여 Gait Heading Yaw PI 보정값을 만든다.

$$
\omega_{z,heading,feedback}
=
PI_\psi
\left(
\operatorname{wrap}_{\pi}(\psi_{ref}-\psi_{IMU})
\right)
$$

조종 모드 Roll·Pitch와 보정 모드 Yaw는 별도의 단일 자세 PI로 처리한다.

$$
\omega_{posture}
=
PI_R
\left(
\eta_{posture,target}-\eta_{IMU}
\right)
$$

조종 모드 자세 PI에는 Roll·Pitch 오차만 사용하고, 보정 모드 자세 PI에는 $\psi_{corr,target}$과 IMU Yaw의 오차만 사용한다. 내부 각속도 PI는 추가하지 않는다. 자세 PI 출력은 200 Hz로 적분하여 IK 직전 몸체 자세 오버레이에 사용한다.

## Step 10. Final Gait Body Twist

조종 모드에서는 사용자 Feedforward 명령과 x·y Position PI 및 Gait Heading Yaw PI 보정값을 합쳐 실제 보행에 사용할 선속도와 Yaw 각속도를 만든다. 자세 PI 출력은 합하지 않는다.

$$
v_B
=
v_{user}
+
v_{feedback}
$$

$$
\omega_B
=
\omega_{user}
+
\omega_{heading,feedback}
$$

이후 Saturation 및 Rate Limit을 적용한다.

보정 모드에서는 x·y·z 이동속도를 PI 없이 직접 사용하고 보정 Yaw는 자세 오버레이에만 사용한다. 암 모드에는 Position PI와 자세 PI를 적용하지 않는다.

## Step 11. Gait Manager

`Body_Control_Enable=0`이면 보행용 Body Twist와 자세 오버레이를 정지하고 상위 안전·서기·착지 블록의 발끝 목표를 사용한다. `Body_Control_Enable=1`, `Tripod_Enable=0`이면 Tripod 위상을 정지하고 6개 다리의 기본 또는 직전 안전 STANCE 목표를 계속 출력한다. 두 Enable이 모두 1일 때만 현재 보행 위상을 기준으로 각 다리가 STANCE인지 SWING인지 결정하고 위상 전환 조건을 관리한다.

현재 Tripod Phase와 각 다리의

$$
STANCE/SWING
$$

상태를 결정한다.

## Step 12. Stance Trajectory

지면에 닿아 있는 STANCE 발이 몸체 이동에 맞춰 상대적으로 반대 방향으로 움직이도록 다음 발끝 위치를 계산한다.

$$
p_i[k+1]
=
p_i[k]
+
(-v_B-\omega_B\times p_i)T_s
$$

## Step 13. Swing Target

SWING 다리가 다음에 착지해야 할 목표 위치를 현재 몸체 속도와 회전 속도를 이용해 계산한다.

$$
p_{r,i}
=
p_{0,i}
+
\frac{T_{stance}}{2}
(v_B+\omega_B\times p_{0,i})
$$

## Step 14. Swing Bezier and Radial Offset

현재 발끝 위치에서 착지 목표점까지 부드럽게 이동하도록 Bezier 곡선을 생성하고, Swing 중간에는 다리 바깥 방향의 방사 오프셋을 추가한다.

$$
p_i^{ref}
=
B(s)
+
4r_{swing}s(1-s)e_{r,i}
$$

## Step 15. Early / Late Landing

예상보다 일찍 닿거나 늦게 닿는 상황을 압력센서로 판단해 STANCE 전환 또는 Search Down을 수행한다.

압력센서 접촉 상태에 따라 발끝 명령과 다리 상태를 수정한다.

Early Landing이 검출되면 현재 발끝 위치를 Stance 시작 위치로 저장하고 해당 다리를 즉시

$$
State_i=STANCE
$$

로 전환한다. 같은 제어 주기에서는 접촉 위치를 유지하고, 다음 제어 주기부터 Stance Trajectory를 적용한다.

Swing 종료 후에도 접촉하지 않은 다리는

$$
State_i=LATE\_LANDING
$$

으로 전환하여 $-z_B$ 방향으로 내리는 동시에 각 다리 장착점 방향으로 당기는 Search Down을 수행한다.

## Step 16. Body Posture Overlay

Foot Target Selector가 선택한 STANCE Hold 또는 Tripod 기준 발끝 목표에 단일 자세 PI가 만든 몸체 자세의 역회전을 적용한다.

$$
{}^Bp_{F_i}^{cmd}
=
({}^RR_B(\eta_{posture,cmd}))^T
{}^Rp_{F_i}^{base}
$$

이 단계는 보행 위상, 보폭, Swing 착지점을 변경하지 않고 몸체 원점 기준 자세만 변경한다.

## Step 17. Coordinate Transform

자세 오버레이로 몸체 좌표계에 변환된 발끝 목표 위치를 각 다리의 로컬 좌표계로 변환한다.

$$
{}^{L_i}p_F^{ref}
=
({}^BR_{L_i})^T
(
{}^Bp_F^{cmd}
-
{}^Bp_{L_i}
)
$$

## Step 18. Inverse Kinematics

각 다리 로컬 좌표계의 발끝 목표 위치를 실제 3개 관절의 목표각으로 변환한다.

$$
{}^{L_i}p_F^{ref}
\xrightarrow{IK}
q_i^{ref}
$$

## Step 19. Safety

계산된 자세·발끝 위치·관절각이 안전 범위를 벗어나는지 검사하고 필요한 제한 또는 정지를 적용한다.

순서대로

1. 전복 상태 검사
2. IK Workspace 검사
3. 관절 범위 검사
4. 지지다각형 검사
5. 비정상 목표각 Jump 검사
6. 최대 관절 각속도 제한

을 수행한다.

전복 상태 검사는 다음 조건으로 판단하며, 조건을 만족하면 즉시 로봇을 정지한다.

$$
|\phi|\ge80^\circ
\quad\lor\quad
|\theta|\ge80^\circ
$$

## Step 20. Servo Output

최종 관절 명령각을 DS51150-270 서보용 PWM Pulse Width로 변환하여 18개 서보에 출력한다.

$$
q^{cmd}
\rightarrow
Servo
$$

---





# 31. 권장 초기 튜닝 순서

제어기는 다음 순서로 튜닝하는 것이 안전하다.

## 31.1 기구학 검증

먼저 PI를 모두 끄고

$$
p^{ref}
\rightarrow IK
\rightarrow Servo
$$

만 검증한다.

확인 항목:

- 각 다리 좌표축 방향
- 관절 양/음 방향
- Servo Zero Offset
- FK와 IK 상호 일치
- 관절 제한

---

## 31.2 Stance 궤적 검증

Feedback을 끈 상태에서 낮은 속도로

$$
\dot p_i
=
-v-\omega\times p_i
$$

만 적용한다.

전진 명령 시 Stance 발끝이 몸체 좌표계에서 뒤쪽으로 이동하는지 확인한다.

---

## 31.3 Contact Detection

$F_{contact}$, $F_{release}$, $N_c$, $N_r$를 먼저 튜닝한다.

정지 상태에서 접촉과 미접촉이 안정적으로 구분되어야 한다.

---

## 31.4 Body Position PI

Position PI는 조종 모드의 절대좌표계 x·y 위치에 대해서만 튜닝한다. z축과 보정 모드 x·y·z 이동속도에는 적용하지 않는다.

처음에는

$$
K_I=0
$$

으로 두고 $K_P$만 증가시킨다.

몸체 위치가 충분히 복원되면서 과도한 진동이 발생하지 않는 수준의 $K_P$를 먼저 찾는다.

그 후 정상상태 오차가 남는 경우에만 작은 $K_I$를 추가한다.

적분항이 너무 크면 느린 진동이나 과도한 보정이 발생할 수 있으므로 $K_I$는 작게 시작한다.

---

## 31.5 Body Attitude PI

조종 모드의 Roll / Pitch를 먼저 튜닝한다.

권장 순서는

$$
P
\rightarrow
I
$$

이다.

먼저 $K_P$를 조정하여 자세 복원 성능을 확보하고, 정상상태 자세 오차가 남는 경우에만 작은 $K_I$를 추가한다.

조종 모드 Yaw는 Gait Heading Hold PI로 먼저 별도 튜닝한다. 보정 모드 Yaw는 Roll·Pitch와 같은 단일 몸체 자세 PI 구조를 사용하되 직접 목표각으로 처리한다. 두 Yaw PI는 목적과 출력 경로가 다르므로 Gain과 적분 상태를 분리하고, 모드 전환 시 목표각과 적분항을 각각 초기화한다. 내부 각속도 PI는 추가하지 않는다.

---

## 31.6 Gait

몸체 안정화가 확인된 후

1. 낮은 $V_{x,\max}$
2. 긴 $T_{phase}$
3. 높은 안정 여유
4. 작은 Step Length

에서 시작하고 점차 속도를 증가시킨다.

---

# 32. 주요 튜닝 파라미터

| 구분 | 파라미터 | 의미 |
|---|---|---|
| 제어 | $f_s$ | 제어 주파수, 200 Hz |
| 제어 | $T_s$ | 제어 주기, 0.005 s |
| RC | $\delta$ | 조이스틱 Dead Zone |
| RC | $f_c$ | 입력 LPF Cutoff |
| RC | $V_{x,\max}$ | 최대 전진 속도 |
| RC | $\Omega_{z,\max}$ | 최대 Yaw 각속도 |
| RC / 자세 | $\phi_{\max}$ | 최대 Roll 목표각 |
| RC / 자세 | $\theta_{\max}$ | 최대 Pitch 목표각 |
| 보정 | $\psi_{corr,max}$ | 최대 Yaw 보정 목표각 |
| Position PI | $K_{P,p}^{xy}, K_{I,p}^{xy}$ | 조종 모드 x·y 위치 PI Gain |
| Heading Yaw PI | $K_{P,\psi,h}, K_{I,\psi,h}$ | 조종 모드 보행 Heading PI Gain |
| Posture PI | $K_{P,R}, K_{I,R}$ | 조종 모드 Roll·Pitch 및 보정 모드 Yaw 단일 자세 PI Gain |
| Feedback | $v_{fb,\max}^{xy}$ | 최대 x·y 위치 보정 속도 |
| Feedback | $\omega_{heading,fb,max}$ | 최대 보행 Heading 보정 각속도 |
| Feedback | $\omega_{posture,max}$ | 최대 몸체 자세 보정 각속도 |
| Gait | $T_{phase}$ | Tripod Phase 시간 |
| Gait | $T_{stance}$ | Stance 예상 시간 |
| Swing | $T_{swing}$ | Swing 시간 |
| Swing | $h_0$ | 기본 Swing Height |
| Swing | $r_{swing}$ | Swing 최고점의 방사 방향 오프셋, 현재 0.07 m |
| Swing | $h_{\min}, h_{\max}$ | Swing Height 제한 |
| Contact | $F_{contact}$ | 접촉 Threshold |
| Contact | $F_{release}$ | 해제 Threshold |
| Contact | $N_c, N_r$ | 연속 판정 Sample 수, 200 Hz 기준 |
| Early | $s_{early,\min}$ | Early Landing 검사 시작 구간 |
| Late | $v_{search}$ | 하강 탐색 속도, 현재 0.20 m/s |
| Late | $k_{in}$ | 하강 속도 대비 안쪽 이동 비율, 현재 0.8 |
| Late | $v_{in}$ | 안쪽 이동 속도, $k_{in}v_{search}$, 현재 0.16 m/s |
| Late | $d_{search,\max}$ | 최대 대각 탐색 거리 |
| Late | $T_{search,\max}$ | 최대 탐색 시간 |
| Joint | $\dot{\theta}_{\max}$ | 최대 관절 각속도 |
| Joint | $\Delta\theta_{jump,\max}$ | 비정상 목표각 Jump 기준 |
| Stability | $k_{support}=0.9$ | 안전 지지다각형 선형 Scale Factor |
| Safety | $\phi_{rollover}=80^\circ$ | Roll 전복 판정 기준 |
| Safety | $\theta_{rollover}=80^\circ$ | Pitch 전복 판정 기준 |
| Estimator | $r_{slip,\max}$ | Stance Foot Slip Reject 기준 |

---

# 33. 최종 제어 구조 요약

전체 제어기는 다음 식으로 요약할 수 있다.

사용자 명령:

$$
\begin{aligned}
v_{user},\;\phi_{ref},\;\theta_{ref},\;\omega_{z,user}
&= f_{RC}(u),\\
\omega_{user}
&=
\begin{bmatrix}
0 & 0 & \omega_{z,user}
\end{bmatrix}^{T}
\end{aligned}
$$

조종 모드의 몸체 x·y 목표 위치:

$$
\boxed{
p_{B,ref}^{xy}[k]
=
p_{B,ref}^{xy}[k-1]
+
\begin{bmatrix}
\cos\psi_{ref}[k]\\
\sin\psi_{ref}[k]
\end{bmatrix}
v_{x,user}[k]T_s
}
$$

Position Reference Generator 활성화 시에는 $p_{B,ref}^{xy}[k_0]={}^W\hat p_B^{xy}[k_0]$로 초기화한다.

몸체 위치 Feedback:

$$
v_{feedback}^{xy}
=
PI_p
\left(
p_{B,ref}^{xy}-{}^W\hat p_B^{xy}
\right)
$$

조종 모드의 Gait Heading Yaw Feedback:

$$
\omega_{z,heading,feedback}
=
PI_{\psi,h}
\left(
\operatorname{wrap}_{\pi}(\psi_{ref}-\psi_{IMU})
\right)
$$

조종 모드 Roll·Pitch 및 보정 모드 Yaw 단일 자세 PI:

$$
\omega_{posture}
=
PI_R
\left(
\eta_{posture,target}-\eta_{IMU}
\right)
$$

Final Gait Body Twist:

$$
\boxed{
\xi_{gait}
=
\begin{bmatrix}
v_{user}+v_{feedback}\\
0\\
0\\
\omega_{z,user}+\omega_{z,heading,feedback}
\end{bmatrix}
}
$$

Stance:

$$
\boxed{
\dot p_i
=
-v_B-\omega_B\times p_i
}
$$

Swing 착지점:

$$
\boxed{
p_{r,i}
=
p_{0,i}
+
\frac{T_{stance}}{2}
\left(
v_B+\omega_B\times p_{0,i}
\right)
}
$$

Swing 궤적:

$$
\boxed{
p_i^{ref}
=
Bezier
(
P_0,P_1,P_2,P_3,s
)
+
4r_{swing}s(1-s)e_{r,i}
}
$$

위 합산식은 조종 모드의 보행 경로에 적용한다. 보정 모드는 x·y·z 이동속도를 PI 없이 직접 사용하고 보정 Yaw 자세 PI는 보행 Body Twist에 합하지 않는다. 암 모드에는 Position PI와 자세 PI를 적용하지 않는다.

발 접촉 보정:

$$
p_i^{ref}
\rightarrow
Early/LateLanding
$$

기준 발끝 목표 선택:

$$
{}^Rp_i^{base}
=
\begin{cases}
{}^Rp_i^{tripod}, & Tripod\_Enable=1\\
{}^Rp_i^{stance\_hold}, & Tripod\_Enable=0
\end{cases}
$$

`Tripod_Enable=0`일 때도 $p_i^{stance\_hold}$를 매 주기 출력한다.

몸체 자세 오버레이:

$$
{}^Bp_i^{cmd}
=
({}^RR_B(\eta_{posture,cmd}))^T
{}^Rp_i^{base}
$$

자세 오버레이는 Tripod 위상, 보폭, Stance/Swing 궤적과 착지점을 변경하지 않는다.

좌표 변환:

$$
{}^Bp_i^{cmd}
\rightarrow
{}^{L_i}p_i^{ref}
$$

역기구학:

$$
{}^{L_i}p_i^{ref}
\rightarrow
q_i^{ref}
$$

안전장치:

$$
q_i^{ref}
\rightarrow
RolloverCheck
\rightarrow
Workspace
\rightarrow
JointLimit
\rightarrow
JumpCheck
\rightarrow
RateLimit
$$

최종 출력:

$$
\boxed{
q^{cmd}
\rightarrow Servo
}
$$

이 구조를 매 $0.005\,\mathrm{s}$마다 반복한다.
