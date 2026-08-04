# Hexapod-Robot MuJoCo MJX 완전 가이드

이 문서는 **지금 네 컴퓨터 상태에 맞춰서**, Hexapod-Robot의 MuJoCo MJX 학습 환경을 **장기적으로 덜 깨지고, 재현 가능하고, 디버깅하기 쉬운 방식**으로 정리한 문서다.

목표는 4개다.

1. **안정적인 GPU 학습 환경**을 유지한다.
2. **MuJoCo MJX로 기초 보행을 먼저 잡는다.**
3. 그 다음 **Isaac Sim / Isaac Lab 강화학습으로 확장**한다.
4. 나중에 네가 혼자 봐도 **구조, 변경 포인트, 디버깅 포인트, 공부 순서**를 바로 이해할 수 있게 한다.

---

## 0. 이번에 최종 정리한 결론

### 권장 운영 방식

- **JAX/MJX 전용 환경은 pip bundled CUDA 방식으로 격리 유지**
- **시스템 전체 CUDA toolkit은 JAX와 분리해서 생각**
- JAX 실행은 항상 `LD_LIBRARY_PATH` 영향을 제거한 wrapper로 실행
- 로컬에 CUDA 12.8 / 13.2 흔적이 있어도, **JAX는 그걸 직접 믿지 않고 자기 wheel 조합으로 실행**

### 최종 고정 버전

- Python: **3.10.12**
- Driver: **580.173.02**
- GPU: **RTX 3090**
- JAX: **0.6.2**
- jaxlib: **0.6.2**
- jax-cuda12-plugin: **0.6.2**
- jax-cuda12-pjrt: **0.6.2**
- MuJoCo: **3.10.0**
- mujoco-mjx: **3.10.0**

### 왜 CUDA 13이 아니라 CUDA 12 계열 JAX로 갔나

지금 이 머신의 Python 3.10 기준으로 실제 설치/검증이 깔끔한 조합이 **JAX 0.6.2 + cuda12 plugin/pjrt** 였다.

즉,

- 드라이버는 580이라 충분히 최신
- 하지만 **현재 Python/JAX 릴리스 조합에서 가장 덜 꼬이는 GPU 경로는 cuda12 쪽**
- 장기적으로도 이 조합이 **실전에서 덜 깨진다**

이건 “시스템에 CUDA 13 폴더가 있으니 무조건 13을 써야 한다”가 아니라,
**JAX wheel과 Python 버전이 실제로 가장 안정적으로 맞는 조합을 택한 것**이다.

---

## 1. 지금 컴퓨터에서 확인된 상태

### GPU/드라이버

- NVIDIA GeForce RTX 3090
- Driver Version: 580.173.02

### CUDA 관련 시스템 상태

- `/usr/local/cuda -> /etc/alternatives/cuda`
- `/usr/local/cuda-12.8` 존재
- `/usr/local/cuda-13.2` 존재
- `nvcc`는 기본 PATH에 없음
- system-wide cuDNN / NCCL은 JAX가 자동으로 신뢰할 만큼 깔끔하게 노출되지 않음

즉, 시스템은 **툴킷이 약간 섞여 있는 상태**고,
JAX 입장에서는 이런 상태를 그냥 믿고 타는 것보다 **자기 wheel CUDA 조합을 쓰는 게 더 안전**하다.

---

## 2. 왜 “툴킷 있는 게 무조건 더 좋은 것”은 아닌가

이건 헷갈리기 쉬운데, 구분하면 쉽다.

### A. 시스템 CUDA toolkit이 필요한 경우

이런 작업에는 toolkit이 있으면 좋다.

- `nvcc`로 CUDA 코드 직접 컴파일
- PyTorch CUDA extension 빌드
- custom CUDA kernel 개발
- TensorRT
- Nsight profile / debug
- C++/CUDA 연구 코드

### B. JAX / MJX만 안정적으로 돌릴 때

이 경우는 toolkit보다 **버전 일치와 격리**가 더 중요하다.

JAX가 가장 자주 깨지는 이유는 보통 아래다.

- `LD_LIBRARY_PATH`가 이상한 CUDA를 가리킴
- toolkit 버전과 cuDNN/NCCL이 안 맞음
- PATH에 여러 CUDA가 섞임
- 로컬 설치 CUDA와 pip wheel CUDA가 충돌

그래서 **JAX 전용 환경은 toolkit 의존도를 줄이는 게 오히려 더 안정적**이다.

### 장기적으로 제일 좋은 방식

둘 다 가지면 된다.

- **JAX/MJX 전용 isolated venv**: 학습/실험용 안정 환경
- **시스템 toolkit 1개 정리**: `nvcc`, profiling, 다른 프레임워크 개발용

이번 세팅은 우선 **JAX/MJX 안정 환경**을 먼저 만든 것이다.

---

## 3. 이번에 다시 정리한 세팅

### 정리한 것

이전 MJX 세팅 흔적은 지우고 다시 만들었다.

- `~/.venvs/hexapod-mjx` 재생성
- `~/bin/hexapod-mjx-python` 재생성
- `~/bin/hexapod-mjx-train` 재생성
- repo 내부 `SW/mjx/artifacts`, `__pycache__` 정리

### 현재 유효한 실행 진입점

#### 1) Python wrapper

```bash
hexapod-mjx-python
```

이 wrapper는 아래를 보장한다.

- `~/.venvs/hexapod-mjx/bin/python` 사용
- `LD_LIBRARY_PATH` 제거
- 실수로 system python / 섞인 CUDA를 타는 상황 방지

#### 2) 학습 wrapper

```bash
hexapod-mjx-train
```

이 wrapper는 아래를 보장한다.

- `~/Hexapod-Robot` repo로 이동
- 안정 venv python 사용
- `SW/mjx/train_tripod_cem.py` 실행

---

## 4. 네가 지금 바로 써야 하는 명령

### 환경 확인
```bash
hexapod-mjx-python - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
PY
```

정상 기대값:
- backend = `gpu`
- devices = `[CudaDevice(id=0)]`

### GPU 스모크 테스트
```bash
hexapod-mjx-train \
  --population-size 8 \
  --elite-count 2 \
  --num-iterations 1 \
  --rollout-steps 60 \
  --action-repeat 2 \
  --seed 0 \
  --output-path SW/mjx/artifacts/hexapod_cem_gpu_smoke.json
```

이건 아주 작은 population으로 “GPU backend + MJX + repo 코드가 최소한 정상 작동하는지” 확인하는 용도다.

### 기본 학습
```bash
hexapod-mjx-train \
  --population-size 16 \
  --elite-count 4 \
  --num-iterations 2 \
  --rollout-steps 120 \
  --action-repeat 2 \
  --seed 0 \
  --output-path SW/mjx/artifacts/hexapod_cem_gpu_baseline.json
```

### 결과 시각화
```bash
env -u LD_LIBRARY_PATH ~/.venvs/hexapod-mjx/bin/python \
  SW/mjx/visualize_tripod_gait.py \
  --result-path SW/mjx/artifacts/hexapod_cem_gpu_baseline.json
```

참고:

- `DISPLAY` 가 있는 데스크톱 세션이면 viewer가 열린다.
- `DISPLAY` 가 없는 tmux/SSH/headless 세션이면 이 wrapper는 자동으로 EGL headless frame export로 fallback 한다.

viewer를 **실제로 띄우는 방법**은 아래처럼 보면 된다.

1. **가장 확실한 방법: 이 PC의 데스크톱 터미널에서 실행**

   - Ubuntu GUI에서 터미널을 직접 열고 실행하면 된다.
   - 이 경우 `DISPLAY` 가 이미 잡혀 있으므로 별도 설정이 거의 필요 없다.

   ```bash
   echo "$DISPLAY"
   echo "$XAUTHORITY"
   env -u LD_LIBRARY_PATH ~/.venvs/hexapod-mjx/bin/python SW/mjx/visualize_tripod_gait.py --result-path SW/mjx/artifacts/hexapod_cem_gpu_baseline.json
   ```

2. **tmux를 꼭 써야 하면, GUI 터미널에서 tmux를 붙는 방식이 가장 안전**

   - 데스크톱 터미널에서 `tmux attach` 해서 같은 세션으로 들어가면 `DISPLAY` 를 그대로 물고 간다.
   - 이미 떠 있는 headless tmux 세션이라면, GUI 터미널에서 확인한 값을 복사해서 넣어야 한다.

   ```bash
   export DISPLAY=:0
   export XAUTHORITY=$HOME/.Xauthority
   env -u LD_LIBRARY_PATH ~/.venvs/hexapod-mjx/bin/python SW/mjx/visualize_tripod_gait.py --result-path SW/mjx/artifacts/hexapod_cem_gpu_baseline.json
   ```

   여기서 `:0` 는 예시다. **반드시 GUI 터미널에서 `echo $DISPLAY` 로 실제 값 확인 후** 넣는 게 맞다.

3. **SSH / headless 세션이면 viewer에 집착하지 말고 headless로 보는 게 맞다**

   - 지금처럼 `DISPLAY` 가 비어 있으면 GLFW viewer는 안 뜬다.
   - 이 가이드의 wrapper들은 그런 경우 자동으로 EGL headless frame export로 fallback 한다.
   - 즉, 원격 세션에서는 보통 `.ppm` 프레임 저장 경로를 보는 쪽이 정석이다.

이 스크립트는 가장 최근 학습 JSON의 `best_params`를 읽어서,
**학습 때 쓴 open-loop tripod gait를 MuJoCo viewer에서 다시 재생**한다.

직접 실행하면:

```bash
cd ~/Hexapod-Robot
env -u LD_LIBRARY_PATH ~/.venvs/hexapod-mjx/bin/python \
  SW/mjx/visualize_tripod_gait.py \
  --result-path SW/mjx/artifacts/hexapod_cem_gpu_baseline.json
```

이때 내부적으로 하는 일:

- 저장된 `best_params` 로드
- 같은 standing pose / 같은 joint order 사용
- 학습 때와 같은 PD 제어기 + `action_repeat` 로 다시 재생
- 즉 “JSON 숫자가 실제로 어떤 보행으로 보이는지”를 눈으로 확인

GUI viewer를 못 띄우는 환경이면 프레임 저장으로 확인할 수 있다.

```bash
cd ~/Hexapod-Robot
MUJOCO_GL=egl env -u LD_LIBRARY_PATH ~/.venvs/hexapod-mjx/bin/python \
  SW/mjx/visualize_tripod_gait.py \
  --result-path SW/mjx/artifacts/hexapod_cem_gpu_baseline.json \
  --render-dir SW/mjx/artifacts/visualize_baseline_frames \
  --duration-sec 4
```

이 명령은 `.ppm` 프레임들을 저장한다.
추가 라이브러리 없이도 남길 수 있게 일부러 단순 포맷을 쓴 것이다.

### best score에서 이어서 학습 + 바로 시각화

```bash
cd ~/Hexapod-Robot
env -u LD_LIBRARY_PATH ~/.venvs/hexapod-mjx/bin/python \
  SW/mjx/train_tripod_cem.py \
  --resume-result-path SW/mjx/artifacts/hexapod_cem_gpu_baseline.json \
  --output-path SW/mjx/artifacts/hexapod_cem_gpu_baseline.json \
  --population-size 32 \
  --elite-count 4 \
  --num-iterations 3 \
  --rollout-steps 180
```

이 흐름은 기본적으로:
- 기존 `SW/mjx/artifacts/hexapod_cem_gpu_baseline.json` 이 있으면 그 `best_params`에서 이어서 탐색
- 새 best를 다시 JSON에 저장
- 저장 직후 MuJoCo viewer나 headless render로 다시 확인

헤드리스 렌더 예시:
```bash
cd ~/Hexapod-Robot
env -u LD_LIBRARY_PATH MUJOCO_GL=egl ~/.venvs/hexapod-mjx/bin/python \
  SW/mjx/visualize_tripod_gait.py \
  --result-path SW/mjx/artifacts/hexapod_cem_gpu_baseline.json \
  --render-dir SW/mjx/artifacts/hexapod_cem_gpu_baseline_frames \
  --duration-sec 4
```

### Residual RL 학습
이제 MJX 쪽에는 **고전 제어기 + residual RL 정책** 경로도 추가되어 있다.

핵심 아이디어는:
- tripod gait, command filtering, safety projection, joint-space 제어는 명시적 로직
- RL은 작은 action space로 **착지/보폭/몸체 높이/roll-pitch trim**만 보정
- 즉, 정책이 18개 관절 전체를 직접 내뱉지 않는다

학습/시각화는 이제 번호형 스크립트 대신 아래 3개 단축 명령을 쓴다.
- `~/Desktop/Hexapod-MJX-가이드/빠른학습.sh`
- `~/Desktop/Hexapod-MJX-가이드/시각화.sh`
- `~/Desktop/Hexapod-MJX-가이드/큰병렬.sh`

이 wrapper는 이제 기본적으로 아래를 한 번에 한다.

- residual RL 학습 진행
- 기존 checkpoint가 있으면 그 latest checkpoint에서 이어서 학습
- 학습 중 **가장 높은 mean_reward를 만든 policy는 즉시 best checkpoint로 저장**
- 학습 중 **현재 optimizer state는 latest checkpoint로 계속 저장**
- metrics JSON도 update마다 계속 다시 저장
- 그 best policy를 바로 headless 렌더해서 **MP4 하나만 저장**
- MP4는 시작하자마자 바로 policy를 먹이지 않고 **neutral pose를 약 1.5초 먼저 보여준 뒤** rollout을 시작
- 학습 환경에서는 로봇의 **등/몸통 collision box가 바닥에 닿을 때만 done 처리**하고, 기본 threshold도 예전보다 덜 빡세게 `0.0` 기준으로 맞춰져 있다
- 그리고 매 실행마다 **`SW/mjx/artifacts/residual_rl_runs/YYYYMMDD/<run-stem>/`** 폴더를 새로 만들어 산출물을 그 안에 따로 저장

즉, 예전처럼 frame 폴더를 기본 산출물로 남기지 않고,
**최고 점수 정책의 MP4만 바로 남기는 흐름**으로 바뀌었다.

기본 저장 경로:

- 날짜 폴더: `SW/mjx/artifacts/residual_rl_runs/YYYYMMDD/`
- run 폴더: `SW/mjx/artifacts/residual_rl_runs/YYYYMMDD/<run-stem>/`
- best policy checkpoint: `<run-folder>/<run-stem>.pkl`
- latest checkpoint: `<run-folder>/<run-stem>_latest.pkl`
- metrics JSON: `<run-folder>/<run-stem>_metrics.json`
- run metadata JSON: `<run-folder>/<run-stem>_run.json`
- best-policy MP4: `<run-folder>/<run-stem>.mp4`

`_run.json` 안에는 아래가 남는다.

- 실행 날짜
- 실제 실행 명령어
- `num_envs`, `rollout_steps`, `num_updates`, `ppo_epochs`, `minibatch_size`, `hidden_size`, `learning_rate`, `seed`
- checkpoint / metrics / mp4 실제 저장 경로
- 실제 run output 폴더 경로
- 가능하면 마지막 best score 요약

추가 옵션:

- `--fresh` : 기존 checkpoint를 무시하고 처음부터 다시 학습
- `--skip-train` : 학습은 건너뛰고 현재 checkpoint만 MP4로 저장
- `--output-video <path>` : MP4 저장 경로 직접 지정
- `--run-label <name>` : 파일명과 run 폴더 이름에 들어갈 라벨 지정

`--fresh` 로 시작하면 그 run의 **neutral pose PNG + JSON** 도 같은 run 폴더에 따로 저장된다.
- `<run-folder>/<run-stem>_neutral_pose.png`
- `<run-folder>/<run-stem>_neutral_pose.json`
### W&B 연동

이제 코드는 **W&B를 실제로 붙일 수 있게 연결돼 있다.**

동작 방식:

1. 학습 시작 시 W&B run open
2. PPO update마다 scalar 기록
   - `mean_reward`
   - `best_mean_reward`
   - `actor_loss`
   - `value_loss`
   - `done_rate`
   - 속도/자세 관련 metric
3. 학습 중 로컬 파일은 계속 저장
   - best checkpoint
   - latest checkpoint
   - metrics json
4. 학습 종료 후
   - best/latest checkpoint
   - metrics json
   - run json
   - mp4
   를 같은 W&B run에 artifact로 업로드

즉 구조는 다음과 같다.

- **로컬 파일이 기준**
- **W&B는 그래프/비교/공유 계층**
- MuJoCo/MJX와 Isaac Sim은 한 환경에 억지로 합치지 않고, 실험 기록만 W&B에서 묶는다.

### 설치

```bash
~/.venvs/hexapod-mjx/bin/python -m pip install wandb==0.20.1
```

또는 가이드 폴더의 `residual_rl_run.sh` / `빠른학습.sh` / `큰병렬.sh` 를 사용한다.

### 로그인

```bash
~/.venvs/hexapod-mjx/bin/python -m wandb login
```

### 기본 사용법

```bash
~/Desktop/Hexapod-MJX-가이드/큰병렬.sh fresh \
  --wandb \
  --wandb-project hexapod-residual-rl \
  --wandb-group mjx-baseline \
  --num-envs 256 \
  --rollout-steps 256 \
  --num-updates 5000
```

### 자주 쓰는 W&B 옵션

- `--wandb` : W&B 활성화
- `--wandb-project <name>` : project 이름
- `--wandb-entity <name>` : team / account 이름
- `--wandb-group <name>` : 실험 묶음 이름
- `--wandb-job-type <name>` : 기본값 `mjx-train`
- `--wandb-mode <online|offline|disabled>` : 온라인/오프라인 모드
- `--wandb-tags a,b,c` : 태그
- `--wandb-name <name>` : run 이름 강제 지정

추천 naming:
- project: `hexapod-residual-rl`
- group: `mjx-baseline` 또는 `isaaclab-transfer`
- run name: 기본값으로 현재 `run stem` 사용

### W&B + 빠른 테스트 예시

```bash
~/Desktop/Hexapod-MJX-가이드/빠른학습.sh fresh \
  --wandb \
  --wandb-mode offline \
  --wandb-group mjx-smoke
```

### 오프라인 테스트
로그인 없이 구조만 확인하려면:

```bash
~/Desktop/Hexapod-MJX-가이드/빠른학습.sh fresh \
  --wandb \
  --wandb-mode offline
```

### 주의

- 실행 인자는 **가급적 축약형이 아니라 정식 이름**을 쓴다.
- 즉 `--num-envs`, `--num-updates` 같이 쓰는 편이 안전하다.
- `--num-env`, `--num-update` 도 이제 동작은 하지만, 기록을 읽을 때는 정식 이름으로 통일하는 편이 덜 헷갈린다.
- 이미 시작된 학습 프로세스에는 W&B를 **중간에 붙일 수 없다.** W&B 기록이 필요하면 그 런은 다시 시작해야 한다.

중간에 `Ctrl+C` 로 끊더라도 현재 update가 끝난 시점 기준으로 아래 파일은 남도록 바꿨다.

- best checkpoint
- latest checkpoint
- metrics json

즉 **장시간 학습을 멈췄다가 다시 이어가기 쉬운 구조**다.

렌더만 다시 보고 싶으면:
```bash
~/Desktop/Hexapod-MJX-가이드/시각화.sh latest --duration-sec 5
```

자주 쓰는 단축 명령은 이제 아래 3개로 보면 된다.

### 1. 빠른학습
```bash
~/Desktop/Hexapod-MJX-가이드/빠른학습.sh [fresh|이어서] [추가 옵션...]
```

예:
```bash
~/Desktop/Hexapod-MJX-가이드/빠른학습.sh fresh
~/Desktop/Hexapod-MJX-가이드/빠른학습.sh 이어서 --num-updates 20 --minibatch-size 32
```

기본 의도:
- 병렬 수를 작게 유지
- rollout/update도 짧게 유지
- smoke 수준으로 빨리 best-policy MP4까지 확인

기본 커리큘럼:
- 1단계: forward-only
- 2단계: forward + small yaw
- 3단계: full command range

### 2. 시각화
```bash
~/Desktop/Hexapod-MJX-가이드/시각화.sh [latest|checkpoint-path] [추가 옵션...]
```

예:
```bash
~/Desktop/Hexapod-MJX-가이드/시각화.sh latest --duration-sec 2
~/Desktop/Hexapod-MJX-가이드/시각화.sh SW/mjx/artifacts/residual_rl_verify.pkl --duration-sec 5
```

설명:
- `latest` 또는 생략: run 폴더 전체를 뒤져서 가장 최근 best checkpoint를 찾아 바로 렌더
- `checkpoint-path`: 특정 checkpoint를 지정해서 렌더
- 내부적으로는 `--skip-train` 모드라 학습은 안 하고 MP4만 만든다
- MP4는 기본적으로 **neutral pose 1.5초 hold + rollout** 순서로 저장된다

### 3. 큰병렬
```bash
~/Desktop/Hexapod-MJX-가이드/큰병렬.sh [fresh|이어서] [추가 옵션...]
```

예:
```bash
~/Desktop/Hexapod-MJX-가이드/큰병렬.sh fresh --num-updates 500
~/Desktop/Hexapod-MJX-가이드/큰병렬.sh 이어서 --num-envs 256 --rollout-steps 256 --minibatch-size 512
```

기본 의도:
- 더 큰 병렬량으로 본격 학습
- 기본 preset은 `--num-envs 96`, `--rollout-steps 96`, `--minibatch-size 512`

### 이어서 / fresh 동작
- `fresh` : 기존 checkpoint를 무시하고 처음부터 다시 시작
- `이어서` : 같은 라벨 계열에서 가장 최근 checkpoint를 자동으로 찾아 이어서 학습
- 새 실행은 매번 **새 날짜/run 폴더**를 만들고, 이어서여도 새 폴더에 현재 런 결과를 저장한다
- 추가 옵션은 뒤에 그대로 붙이면 된다
  - `--num-envs`
  - `--rollout-steps`
  - `--num-updates`
  - `--ppo-epochs`
  - `--minibatch-size`
  - `--hidden-size`
  - `--learning-rate`
  - `--duration-sec`
  - `--termination-contact-z`
  - `--command-curriculum`
  - `--forward-only-updates`
  - `--yaw-stage-updates`
  - `--forward-only-scale`
  - `--yaw-stage-scale`
  - `--wandb`, `--wandb-project`, `--wandb-group`

예를 들어 pose를 바꾼 뒤 정말 처음부터 다시 보고 싶으면:
```bash
~/Desktop/Hexapod-MJX-가이드/큰병렬.sh \
  fresh \
  --wandb \
  --wandb-project hexapod-residual-rl \
  --wandb-group mjx-dome-pose \
  --num-envs 256 \
  --rollout-steps 256 \
  --num-updates 500
```

이렇게 하면 run 폴더에 최소한 아래가 남는다.
- `<stem>.pkl`
- `<stem>_latest.pkl`
- `<stem>_metrics.json`
- `<stem>_run.json`
- `<stem>_neutral_pose.png`
- `<stem>_neutral_pose.json`
- `<stem>.mp4`

예전 번호형 preset도 그대로 남겨뒀지만, 이제는 위 3개 단축 명령을 쓰는 쪽이 덜 헷갈린다.

추가로, 시각화용 모델은 너무 단순해서 다리 움직임이 안 보이던 문제를 줄이기 위해 아래처럼 바꿨다.

- 바닥 plane + 얇은 바닥 slab를 더 크게/밝게 표시
- 몸통 박스는 유지
- 각 다리는 **collision에는 영향 없는 capsule frame** 으로 몸통-다리-발 구체가 이어져 보이게 표시
- 즉 학습용 접촉 단순화는 유지하면서, 보기에는 다리 프레임이 보이게 함


### 환경 재설치

번호형 재설치 스크립트는 제거했다.
이제는 새 venv를 만들고 `requirements-hexapod-mjx-stable.txt` 기준으로 직접 다시 설치하는 방식으로 본다.

---

## 5. 코드 구조 설명: 이 프로젝트에서 MJX가 어떻게 흘러가는가

repo에서 핵심 파일은 4개다.

### `SW/mjx/hexapod_mjx/model.py`

역할:

1. 원본 URDF를 읽는다.
2. `<transmission>`, `<gazebo>` 태그를 제거한다.
3. MuJoCo가 읽을 수 있게 mesh 경로를 정리한다.
4. floating-base MJCF를 만든다.
5. full mesh 충돌을 학습용 단순 충돌로 바꾼다.
6. joint order와 standing pose를 bundle로 만든다.

핵심 아이디어:

- 실제 로봇 CAD/mesh는 **시각적으로는 좋지만 학습용 충돌로는 너무 무겁다.**
- 그래서 MJX 단계에선
  - 몸통: box 1개
  - 발: sphere 6개
  로 단순화했다.

이게 중요한 이유:

- compile 시간이 줄어듦
- rollout 속도가 빨라짐
- collision instability가 줄어듦
- “걷는 구조가 되는지”를 먼저 보는 단계에 맞음

### `SW/mjx/hexapod_mjx/cem.py`

역할:

- tripod gait를 몇 개 파라미터로 표현
- batched rollout으로 여러 후보를 동시에 평가
- CEM(Cross-Entropy Method)로 점수가 좋은 후보 쪽으로 평균/분산 업데이트

이 파일이 사실상 **보행 파라미터 탐색기**다.

여기서 아직 policy network를 학습하는 건 아니다.

### `SW/mjx/train_tripod_cem.py`

역할:

- CLI 진입점
- model bundle 로드
- CEM config 생성
- 결과 출력 및 JSON 저장

즉 이 파일은 “실행 버튼”에 가깝다.

### `SW/mjx/visualize_tripod_gait.py`

역할:

- 학습 결과 JSON의 `best_params`를 읽는다.
- 같은 gait 생성식으로 desired joint target을 다시 만든다.
- MuJoCo viewer 또는 headless frame render로 시각화한다.

즉 이 파일은 “숫자로 저장된 gait를 눈으로 검증하는 버튼”이다.

---

## 6. 지금 학습이 정확히 뭘 하고 있는가

이 단계는 **신경망 정책 학습**이 아니라,
**저차원 open-loop gait parameter search**다.

### 파라미터 공간

현재 탐색하는 파라미터는 대략 이런 느낌이다.

- 보행 주파수
- 첫 번째 관절 진폭
- 두 번째 관절 진폭
- 세 번째 관절 진폭
- hip bias delta
- knee bias delta
- knee phase offset

즉,

> “다리마다 복잡한 정책을 학습”하는 게 아니라,
> “tripod 보행의 기본 파형을 어떤 주기/진폭/위상으로 주면 제일 그럴듯하게 전진하나”
> 를 찾고 있는 것이다.

### 왜 이 접근을 먼저 하냐

강화학습을 바로 돌리면 잘 안 되는 경우 원인이 너무 많다.

- asset 문제
- 충돌 모델 문제
- 초기 자세 문제
- action scaling 문제
- reward 설계 문제
- observation 설계 문제

그런데 MJX + CEM으로 먼저 보면,
최소한 아래를 빨리 확인할 수 있다.

- 이 로봇 구조가 물리적으로 서는가?
- tripod 위상 자체가 말이 되는가?
- joint amplitude가 어느 범위에서 동작하는가?
- 발 contact가 너무 미끄러운가?

즉 **RL 앞단의 기계적 sanity check** 역할을 한다.

---

## 7. 데이터 흐름: 한 번의 학습이 내부에서 어떻게 도는가

1. `train_tripod_cem.py` 실행
2. `load_hexapod_model()` 호출
3. URDF 정리 → floating base MJCF 생성
4. 단순 collision model 생성
5. `mjx.put_model()`로 MJX model 생성
6. CEM이 population만큼 파라미터 샘플링
7. 각 샘플을 batched rollout으로 동시에 평가
8. score 높은 elite 선택
9. elite 평균/표준편차로 다음 분포 갱신
10. best params와 history를 JSON으로 저장

### 출력물

예시:

- `SW/mjx/artifacts/hexapod_floating_base.xml`
- `SW/mjx/artifacts/hexapod_cem_gpu_baseline.json`

JSON에 들어가는 것:

- 실행 config
- best score
- best params
- mean params
- score history
- generated MJCF 경로
- joint order
- 이 JSON은 `SW/mjx/visualize_tripod_gait.py`의 입력으로 바로 재사용 가능

---

## 8. 지금 점수(score)는 무슨 뜻인가

현재 score는 대략 아래 요소를 조합한다.

- 앞으로 간 거리(progress)
- 몸이 얼마나 upright한지
- 횡방향 drift
- yaw 회전
- torque 사용량
- 넘어졌는지 여부

즉,

- **잘 전진하면 점수 증가**
- **옆으로 새거나 많이 비틀리면 감점**
- **토크 너무 많이 쓰면 감점**
- **넘어지면 큰 감점**

이건 “최종 reward”가 아니라 **초기 gait 탐색용 점수 함수**다.

---

## 9. 네가 가장 자주 바꾸게 될 부분

### 9-1. standing pose 바꾸기

파일:

- `SW/mjx/hexapod_mjx/model.py`

대상:

- `STAND_POSE`

이 값은 로봇이 시작할 때의 기본 자세다.

현재 기본값은 **거미처럼 퍼지는 자세가 아니라, 몸통이 위에 있고 다리 링크가 바닥 쪽으로 더 돔처럼 모이도록 다시 조정한 neutral pose** 기준으로 맞춰져 있다.
오른쪽 다리 joint 값 부호가 왼쪽과 반대로 보이는 건 좌우 joint axis가 mirror라서 그렇고, 실제 world-space 자세는 좌우가 같은 방향으로 맞는다.
또한 fresh run은 neutral pose PNG/JSON을 같이 남기고, 시각화 MP4도 시작 1.5초 동안 neutral pose를 먼저 보여주므로 pose 변경이 실제로 반영됐는지 바로 확인할 수 있다.

### neutral pose를 눈으로 튜닝하기

이제 자세만 따로 바로 볼 수 있는 스크립트도 있다.

```bash
~/Desktop/Hexapod-MJX-가이드/자세튜닝.sh --front-q1 -0.7 --mid-q1 0.0 --rear-q1 0.7 --q2 -0.85 --q3 -0.55
```

기본 동작:
- 기본 렌더는 이제 **mesh preview**다. 즉 `~/Downloads/hexa.png` 같은 실물 형상에 가깝게 비교할 수 있다.
- PNG 미리보기 저장: `/tmp/stand_pose_preview.png`
- pose 값 JSON 저장: `/tmp/stand_pose_preview.json`
- 바로 붙여넣기 쉬운 `STAND_POSE` 블록도 같이 출력
- 예전 단순 박스/캡슐 모델로 보고 싶으면 `--visual-style simplified`를 붙이면 된다.

실시간으로 보려면:
```bash
~/Desktop/Hexapod-MJX-가이드/자세튜닝.sh viewer --front-q1 -0.7 --mid-q1 0.0 --rear-q1 0.7 --q2 -0.85 --q3 -0.55
```

자주 쓰는 조절축:
- `--q1` : 몸통 가까운 yaw/sweep 관절 각도
- `--q2` : 가운데 pitch 관절 각도
- `--q3` : 끝 pitch 관절 각도
- `--front-q1/2/3`, `--mid-q1/2/3`, `--rear-q1/2/3` : 앞/중간/뒤 다리만 따로 조절

예:
```bash
~/Desktop/Hexapod-MJX-가이드/자세튜닝.sh --front-q1 -0.7 --mid-q1 0.0 --rear-q1 0.7 --q2 -0.85 --q3 -0.55
~/Desktop/Hexapod-MJX-가이드/자세튜닝.sh --visual-style simplified --front-q1 -0.7 --mid-q1 0.0 --rear-q1 0.7 --q2 -0.85 --q3 -0.55
```

증상별 조정법:
- 시작하자마자 무릎이 접혀서 주저앉음 → `*_2`, `*_3` 기본값 조정
- 다리가 너무 펴져서 발이 떠 있음 → `*_3`를 더 굽힘 쪽으로 조정
- 몸통이 너무 낮음 → stand pose를 먼저 조정하고, reset/replay용 `base_height` 는 pose에서 다시 계산되는지 확인
- 예전 결과 JSON이 옛 pose 기준이면 지금 pose와 1:1 비교하지 말 것

### 9-2. 보행 파라미터 범위 바꾸기

파일:

- `SW/mjx/hexapod_mjx/cem.py`

대상:

- `LOWER_BOUNDS`
- `UPPER_BOUNDS`
- `INITIAL_MEAN`
- `INITIAL_STD`

의미:

- 탐색 가능한 범위
- 처음 어디 근처에서 찾을지
- 처음 얼마나 넓게 퍼뜨릴지

실전 팁:

- 아무것도 안 되는 상태면 범위를 너무 넓히지 말고
  **INITIAL_MEAN**을 더 보수적으로 잡는 게 낫다.
- 어느 정도 걷기 시작하면
  **INITIAL_STD**를 줄여서 refine하는 게 낫다.

### 9-3. PD gain 바꾸기

파일:

- `SW/mjx/hexapod_mjx/cem.py`

대상:

- `PD_KP`
- `PD_KD`
- `TORQUE_LIMIT`

증상별 조정법:

- 발이 후들후들 떨림 → `KD` 증가 고려
- 자세가 너무 흐물함 → `KP` 증가 고려
- 순간적으로 튕기고 불안정 → `KP` 과도 가능성, torque limit도 확인
- 움직임이 너무 답답함 → torque limit 너무 낮을 수 있음

### 9-4. 충돌 모델 바꾸기

파일:

- `SW/mjx/hexapod_mjx/model.py`

현재는:

- base box 1개
- foot sphere 6개

나중에 바꿀 수 있는 것:

- body box 크기
- foot sphere 반지름
- friction 값
- floor friction

이건 보행 품질에 직접 영향 준다.

---

## 10. 어떤 순서로 바꾸는 게 덜 망하나

이 순서가 안전하다.

### 1단계: 시작 자세만 맞추기

- 넘어지지 않고 서는가?
- 발이 바닥에 닿는가?
- 몸통이 너무 낮거나 높지 않은가?

### 2단계: 충돌 단순화 유지

- full mesh collision로 바로 가지 말 것
- base + foot contact만으로 먼저 gait를 잡을 것

### 3단계: gait 파라미터 찾기

- 작은 population, 짧은 rollout으로 smoke test
- 잘 돌면 rollout / iteration / population 확장

### 4단계: 점수 함수 조정

- 전진만 너무 좋아하면 미끄러지면서 달릴 수 있음
- upright / drift / yaw / control cost 균형 조정

### 5단계: 그 다음에야 RL 확장

- Isaac Lab PPO
- observation 설계
- terrain generalization

---

## 11. 학습 파라미터를 어떻게 읽어야 하나

### `population-size`

한 iteration에 평가하는 후보 개수.

- 작으면 빠르지만 탐색이 거칠다.
- 크면 안정적이지만 느리다.

### `elite-count`

상위 몇 개 후보를 다음 분포 업데이트에 쓸지.

- 너무 적으면 운빨이 커짐
- 너무 많으면 업데이트가 둔해짐

### `num-iterations`

몇 번 반복할지.

- smoke test: 1~2
- baseline 탐색: 10 이상
- 제대로 다듬기: 더 늘릴 수 있음

### `rollout-steps`

한 후보를 얼마나 오래 굴릴지.

- 짧으면 빠르지만 “잠깐만 잘하는 후보”가 살아남을 수 있음
- 길면 안정성 판단이 좋아짐

### `action-repeat`

한 번 계산한 target을 몇 simulation step 반복할지.

- 너무 작으면 제어가 예민하고 느려질 수 있음
- 너무 크면 부드럽지만 반응성이 떨어짐

---

## 12. 추천 실험 순서

### A. 매번 먼저 할 것
```bash
hexapod-mjx-python - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
PY
```

### B. smoke test
```bash
hexapod-mjx-train \
  --population-size 8 \
  --elite-count 2 \
  --num-iterations 1 \
  --rollout-steps 60 \
  --action-repeat 2 \
  --seed 0 \
  --output-path SW/mjx/artifacts/hexapod_cem_gpu_smoke.json
```

여기서 확인할 것:
- backend가 GPU인지
- 실행이 끝까지 가는지
- 결과 JSON이 생기는지

### C. baseline 탐색
```bash
hexapod-mjx-train \
  --population-size 16 \
  --elite-count 4 \
  --num-iterations 2 \
  --rollout-steps 120 \
  --action-repeat 2 \
  --seed 0 \
  --output-path SW/mjx/artifacts/hexapod_cem_gpu_baseline.json
```

### D. 더 진지한 탐색 예시

```bash
hexapod-mjx-train \
  --population-size 64 \
  --elite-count 8 \
  --num-iterations 20 \
  --rollout-steps 300 \
  --action-repeat 2 \
  --seed 0 \
  --output-path SW/mjx/artifacts/hexapod_cem_longrun.json
```

처음부터 너무 크게 하지 말고,
**작게 돌아가는 걸 먼저 확인한 뒤** 키워라.

---

## 13. 결과 JSON은 어떻게 읽나

예를 들어 `hexapod_cem_gpu_baseline.json`이 있으면 봐야 할 핵심은:

### `best_score`

현재 탐색 중 가장 좋았던 후보.

### `best_params`

실제로 가장 잘된 gait 파라미터.

이건 나중에 아래 용도로 쓴다.

- 초기 gait priors
- Isaac Lab에서 초기 action bias 참고
- gait generator 초기값 참고

### `mean_params`

마지막 elite 분포의 평균.

- best 하나보다 덜 튀는 값
- 더 안정적인 초기값 후보가 되기도 함

### `score_history`

iteration별 best score 흐름.

- 올라가면 탐색이 되고 있는 것
- 평평하면 탐색 범위/score/pose를 손봐야 함
- 내려가거나 진동하면 elite/pd/rollout 조정 고려

---

## 14. 디버깅 가이드: 어떤 문제가 생기면 어디를 봐야 하나

### 증상 1: GPU가 아니라 CPU로 돌아감

확인:

```bash
hexapod-mjx-python - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
PY
```

정상:

- `gpu`
- `CudaDevice(id=0)`

원인 후보:

- system python으로 실행함
- wrapper 없이 직접 실행함
- `LD_LIBRARY_PATH`가 CUDA를 꼬이게 함
- venv가 망가짐

조치:

1. wrapper로 다시 실행
2. 위의 환경 확인 명령으로 backend/device 확인
3. 안 되면 venv를 지우고 requirements 기준으로 다시 설치

### 증상 2: `ImportError: cannot import name 'mjx' from 'mujoco'`

원인:

- `mujoco`만 있고 `mujoco-mjx`가 없음
- 잘못된 python 사용

조치:

- stable venv 사용
- 재설치 스크립트 실행

### 증상 3: 바로 넘어짐

가장 흔한 원인:

- `STAND_POSE` 부적절
- body/foot collision shape 부적절
- friction 부적절
- PD gain 과도 또는 부족

조치 순서:

1. stand pose
2. foot radius / base box / friction
3. PD gain
4. torque limit

### 증상 4: 전진은 하는데 엄청 미끄러짐

원인 후보:

- friction이 너무 낮음
- score가 progress만 너무 강하게 봄
- torque가 과함
- gait frequency가 너무 높음

조치:

- foot friction 증가
- control cost / yaw / drift penalty 재검토
- 주파수 상한 조정

### 증상 5: rollout은 되는데 score가 계속 안 올라감

원인 후보:

- 탐색 범위가 너무 좁음
- 초기 평균이 너무 나쁨
- elite 수가 너무 작음
- rollout이 너무 짧아 우연한 후보가 살아남음

조치:

- `INITIAL_MEAN` 조정
- `INITIAL_STD` 조정
- `elite-count` 증가
- `rollout-steps` 증가

### 증상 6: 실행이 느리거나 compile이 오래 걸림

원인 후보:

- full mesh collision로 복잡함
- batch가 너무 큼
- rollout이 너무 김
- 처음 JIT compile 중

조치:

- 지금처럼 단순 collision 유지
- 작은 batch로 smoke test 먼저
- 첫 실행 compile 시간과 반복 실행 시간을 분리해서 생각

---

## 15. 꼭 기억해야 할 운영 규칙

### 규칙 1. JAX는 wrapper로 실행

항상 아래 둘 중 하나를 쓴다.

```bash
hexapod-mjx-python
hexapod-mjx-train
```

### 규칙 2. `LD_LIBRARY_PATH`를 임의로 만지지 말 것

JAX/CUDA 문제의 절반은 여기서 시작한다.

### 규칙 3. toolkit을 써도 JAX env와 섞지 말 것

나중에 `nvcc`가 필요하면 **별도 shell / 별도 프로젝트**에서 쓰는 게 낫다.

### 규칙 4. MJX는 “기초 gait 확인용”으로 생각할 것

여기서 최종 locomotion policy를 다 끝내려 하지 말 것.

### 규칙 5. RL 이전에 기계적 sanity를 먼저 확보할 것

- 시작 자세
- 발 contact
- friction
- 토크 크기
- gait frequency

---

## 16. 장기적으로 toolkit을 정리하고 싶다면

현재 시스템에는 CUDA 12.8 / 13.2가 섞여 있다.

### JAX/MJX 기준 추천

그냥 지금처럼 유지:

- JAX는 pip bundled CUDA 사용
- system toolkit은 건드리지 않아도 됨

### 정말 toolkit을 장기적으로 하나로 정리하고 싶다면

추천은 **CUDA 12.8 한 줄기**다.

이유:

- 지금 JAX stable env가 cuda12 계열과 잘 맞음
- nvcc / profiling / extension 개발도 12.8이면 무난함
- 13.2는 지금 JAX 운영상 꼭 필요하지 않음

단,
이건 **시스템 전역 정리 작업**이라 루트 권한과 패키지 정리가 필요하고,
JAX 안정 운영과는 별개다.

즉,

- **학습 안정성**만 보면 지금 세팅이 이미 충분히 좋다.
- **개발 워크스테이션 전체 정리**는 별도 작업이다.

---

## 17. MJX 결과를 Isaac Lab으로 어떻게 이어가나

MJX에서 얻는 것은 보통 3종류다.

1. 기본 stand pose 감각
2. 보행 주파수/진폭 대략 범위
3. contact/frequency가 어느 정도에서 그럴듯한지

이걸 Isaac Lab에 넘길 때는:

- default joint pose 초기값 참고
- action scale 초기 추정에 참고
- reward shaping 방향 참고
- tripod bias를 줄지 말지 판단 참고

### 추천 흐름

1. MJX로 forward-going tripod baseline 확보
2. Isaac Sim에서 asset 정상성 확인
3. Isaac Lab direct task 생성
4. PPO로 command-conditioned locomotion 학습
5. observation 비교 실험
6. terrain 확장

---

## 18. 공부 순서 추천

네가 이 분야를 제대로 이해하려면 아래 순서가 좋다.

### 1단계: MuJoCo / MJX 기본

알아야 할 것:

- `MjModel`, `MjData`
- `qpos`, `qvel`
- free joint
- contact
- rollout
- `jax.jit`, `vmap`, `scan`

추천 학습 포인트:

- “왜 MJX는 batch rollout에 강한가?”
- “왜 physics code를 JAX 함수처럼 다루는가?”

### 2단계: gait engineering

알아야 할 것:

- tripod gait
- stance / swing phase
- phase offset
- duty factor
- stride frequency
- joint bias / amplitude

### 3단계: score / reward 설계

알아야 할 것:

- progress reward
- stability penalty
- control cost
- drift/yaw penalty
- termination penalty

### 4단계: RL 확장

알아야 할 것:

- observation design
- PPO
- reset curriculum
- domain randomization
- terrain generalization

---

## 19. 코드 공부 체크리스트

### model.py 읽을 때

질문:

- 왜 URDF를 그대로 안 쓰고 정리하는가?
- 왜 floating base가 필요한가?
- 왜 full mesh collision를 버렸는가?
- 발 contact point는 어떻게 추출했는가?

### cem.py 읽을 때

질문:

- 보행 파라미터는 몇 차원인가?
- joint target은 time에 따라 어떻게 생성되는가?
- PD torque는 어떻게 계산되는가?
- score는 어떤 요소 합인가?
- elite 업데이트는 어떻게 되는가?

### train_tripod_cem.py 읽을 때

질문:

- 어떤 CLI 인자가 성능/안정성에 영향을 주는가?
- output JSON에 무엇이 저장되는가?

---

## 20. 가장 실용적인 디버깅 루틴

문제 생기면 무조건 이 순서로 간다.

### Step 1. 환경 확인

```bash
hexapod-mjx-python - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
PY
```

### Step 2. 작은 smoke test

```bash
hexapod-mjx-train \
  --population-size 8 \
  --elite-count 2 \
  --num-iterations 1 \
  --rollout-steps 60 \
  --action-repeat 2 \
  --seed 0 \
  --output-path SW/mjx/artifacts/hexapod_cem_gpu_smoke.json
```

### Step 3. 생성된 JSON 확인

- output path 존재 여부
- `best_score`
- `score_history`
- `best_params`

### Step 4. 코드 수정은 한 군데씩만

한 번에 하나만 바꾼다.

- stand pose만
- 또는 friction만
- 또는 bounds만
- 또는 PD gain만

### Step 5. 바꿀 때 로그 남기기

이제 `residual_rl_run.sh`, `빠른학습.sh`, `시각화.sh`, `큰병렬.sh` 는 매 실행마다 `_run.json` 을 자동으로 남긴다.

거기에는 최소한 아래가 들어간다.

- run id / 날짜
- 실제 실행 명령어
- batch / rollout / update / epoch / hidden size / learning rate / seed
- best score 요약
- 산출물 경로
- 해당 실행의 날짜/run 폴더 경로
- fresh 런이면 neutral pose PNG/JSON 경로와 요약

즉 별도 메모를 추가로 남기면 더 좋지만, 최소 실행 기록은 이제 자동 저장된다.

---

## 21. 추천 운영 전략

### 지금 당장

- MJX baseline 계속 다듬기
- 앞으로 가는 tripod gait를 안정화
- best params / mean params 축적

### 그 다음

- Isaac Lab custom task 붙이기
- observation set A baseline
- PPO로 flat terrain locomotion

### 그 후

- contact input 추가
- air-time 추가
- terrain height scan 추가
- rough terrain

---

## 22. 요약

딱 한 줄로 말하면:

> **네 머신에서는 “시스템 CUDA를 믿는 방식”보다 “JAX/MJX 전용 격리 환경 + wrapper 실행”이 더 안전하고, 지금 그 방식으로 다시 정리해둔 상태다.**

그리고 학습 흐름은 이렇게 기억하면 된다.

1. **MJX로 기초 gait 확인**
2. **기계적/충돌/자세 sanity 잡기**
3. **Isaac Lab PPO로 확장**
4. **observation / reward / terrain 일반화로 키우기**

---

## 23. 파일 위치 요약

### 바탕화면 가이드 폴더

- `~/Desktop/Hexapod-MJX-가이드/Hexapod_MJX_완전가이드.md`
- `~/Desktop/Hexapod-MJX-가이드/requirements-hexapod-mjx-stable.txt`
- `~/Desktop/Hexapod-MJX-가이드/residual_rl_run.sh`
- `~/Desktop/Hexapod-MJX-가이드/빠른학습.sh`
- `~/Desktop/Hexapod-MJX-가이드/시각화.sh`
- `~/Desktop/Hexapod-MJX-가이드/큰병렬.sh`

### 실행 wrapper

- `~/bin/hexapod-mjx-python`
- `~/bin/hexapod-mjx-train`

### repo 코드

- `~/Hexapod-Robot/SW/mjx/hexapod_mjx/model.py`
- `~/Hexapod-Robot/SW/mjx/hexapod_mjx/cem.py`
- `~/Hexapod-Robot/SW/mjx/train_tripod_cem.py`
- `~/Hexapod-Robot/SW/mjx/hexapod_mjx/residual_controller.py`
- `~/Hexapod-Robot/SW/mjx/hexapod_mjx/residual_env.py`
- `~/Hexapod-Robot/SW/mjx/hexapod_mjx/residual_rl.py`
- `~/Hexapod-Robot/SW/mjx/train_residual_ppo.py`
- `~/Hexapod-Robot/SW/mjx/evaluate_residual_policy.py`
- `~/Hexapod-Robot/SW/mjx/visualize_residual_policy.py`

---

## 24. 마지막 추천

지금은 **툴킷을 더 만지기보다**, 이 안정 환경에서 MJX gait baseline을 몇 번 반복해서 감을 잡는 게 맞다.

정말 toolkit 기반 개발이 필요해지는 시점은 보통 아래다.

- nvcc로 직접 뭘 빌드해야 할 때
- custom CUDA extension이 필요할 때
- TensorRT / custom kernel 단계로 갈 때

그 전까지는,

- wrapper로 실행
- small smoke test 먼저
- baseline JSON 저장
- 한 번에 하나씩만 수정

이 루틴이 제일 덜 망한다.
