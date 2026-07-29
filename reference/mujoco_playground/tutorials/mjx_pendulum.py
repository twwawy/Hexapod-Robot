from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


XML = r"""
<mujoco model="inverted_pendulum">
  <option timestep="0.002"
          gravity="0 0 -9.81"
          integrator="implicitfast"/>

  <worldbody>
    <light pos="0 -3 3"/>

    <body name="pole" pos="0 0 0">
      <joint name="hinge"
             type="hinge"
             axis="0 1 0"
             damping="0.05"/>

      <geom name="pole_geom"
            type="capsule"
            fromto="0 0 0 0 0 1"
            size="0.05"
            density="500"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="hinge_motor"
           joint="hinge"
           gear="2"
           ctrlrange="-1 1"
           ctrllimited="true"/>
  </actuator>
</mujoco>
"""


# MuJoCo physics는 0.002초 간격으로 적분한다.
# 제어기는 0.02초마다 한 번만 행동을 바꾸고,
# 그 사이에는 같은 action을 유지한다.
SIM_DT = 0.002
CTRL_DT = 0.02
N_SUBSTEPS = int(CTRL_DT / SIM_DT)


# -----------------------------------------------------------------------------
# 모델 준비
# -----------------------------------------------------------------------------
# mj_model: CPU/host 쪽 MuJoCo 모델. viewer/renderer와 함께 쓸 때 필요하다.
# mjx_model: JAX device(GPU/CPU accelerator) 위에서 굴릴 MJX 모델.
mj_model = mujoco.MjModel.from_xml_string(XML)
mjx_model = mjx.put_model(mj_model)


@dataclass(frozen=True)
class ControllerCfg:
    """간단한 PD 기반 안정화 제어기 설정.

    이 튜토리얼의 목적은 RL 학습기가 아니라 MJX step / batched step / 시각화 흐름을
    이해하는 것이다. 그래서 우선은 "여러 번 리셋해도 대충 세워 보이는" 아주 단순한
    제어기를 넣어 두었다.
    """

    kp: float = 0.9
    kd: float = 0.18


def parse_args() -> argparse.Namespace:
    """튜토리얼 실행 방식을 CLI로 고를 수 있게 한다."""
    parser = argparse.ArgumentParser(
        description=(
            "MJX inverted pendulum tutorial. "
            "기본 smoke 실행 + optional viewer 시각화 + batched benchmark를 제공한다."
        )
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="MuJoCo viewer를 열고 여러 에피소드를 눈으로 본다.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="viewer 모드에서 몇 번 reset/rollout 할지 정한다.",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=250,
        help="viewer 모드에서 에피소드 하나당 최대 control step 수.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="reset 샘플링용 시드.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="viewer 모드에서 control dt에 맞춰 sleep하며 사람 눈 속도로 재생한다.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1024,
        help="batched benchmark에 사용할 병렬 환경 수.",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=100,
        help="batched benchmark에서 몇 control step을 굴릴지 정한다.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="시각화만 보고 싶을 때 batched benchmark를 생략한다.",
    )
    return parser.parse_args()


def observe(data: mjx.Data) -> jax.Array:
    """정책에 제공할 관측값을 만든다.

    관측 설계:
    - sin(theta), cos(theta): 각도를 직접 쓰면 2π 주기 경계가 생기므로 삼각함수로 푼다.
    - omega: 각속도.
    """
    theta = data.qpos[0]
    omega = data.qvel[0]

    return jnp.array([
        jnp.sin(theta),
        jnp.cos(theta),
        omega,
    ])


def reset(key: jax.Array) -> tuple[mjx.Data, jax.Array]:
    """하나의 환경을 초기화한다.

    완전히 쓰러진 상태에서 시작하지 않고, upright 근처의 작은 각도/각속도에서 시작한다.
    이렇게 해야 단순 PD 제어만으로도 "세우는 장면"을 쉽게 볼 수 있다.
    """
    key_theta, key_omega = jax.random.split(key)

    theta = jax.random.uniform(
        key_theta,
        shape=(),
        minval=-0.15,
        maxval=0.15,
    )
    omega = jax.random.uniform(
        key_omega,
        shape=(),
        minval=-0.1,
        maxval=0.1,
    )

    data = mjx.make_data(mjx_model)

    qpos = data.qpos.at[0].set(theta)
    qvel = data.qvel.at[0].set(omega)

    data = data.replace(qpos=qpos, qvel=qvel)

    # qpos/qvel만 바꾼 직후에는 site 위치, 관성 관련 파생량이 아직 갱신되지 않았을 수 있으니
    # 한 번 forward를 돌려 물리 상태를 정합시킨다.
    data = mjx.forward(mjx_model, data)

    return data, observe(data)


def step(
    data: mjx.Data,
    action: jax.Array,
) -> tuple[mjx.Data, jax.Array, jax.Array, jax.Array]:
    """한 번의 control step을 실행한다.

    내부에서는 같은 action을 N_SUBSTEPS번 유지하면서 더 작은 physics dt로 적분한다.
    RL 관점에서는 "policy 1step", MuJoCo 관점에서는 "physics 10step"인 셈이다.
    """
    action = jnp.clip(action, -1.0, 1.0)

    def physics_substep(
        current_data: mjx.Data,
        _: None,
    ) -> tuple[mjx.Data, None]:
        current_data = current_data.replace(ctrl=action)
        next_data = mjx.step(mjx_model, current_data)
        return next_data, None

    next_data, _ = jax.lax.scan(
        physics_substep,
        data,
        xs=None,
        length=N_SUBSTEPS,
    )

    theta = next_data.qpos[0]
    omega = next_data.qvel[0]

    # upright(θ=0)에 가까울수록 cos(theta)가 1에 가까워진다.
    upright_reward = jnp.cos(theta)

    # 지나치게 빠른 회전 / 과한 토크는 감점한다.
    velocity_penalty = 0.01 * omega**2
    action_penalty = 0.001 * jnp.sum(action**2)
    reward = upright_reward - velocity_penalty - action_penalty

    invalid_state = (
        jnp.isnan(next_data.qpos).any()
        | jnp.isnan(next_data.qvel).any()
    )

    # 90도 이상 기울어지면 사실상 실패로 간주한다.
    fallen = jnp.abs(theta) > jnp.pi / 2

    done = (invalid_state | fallen).astype(jnp.float32)
    obs = observe(next_data)

    return next_data, obs, reward, done


def controller_action(obs: jax.Array, cfg: ControllerCfg) -> jax.Array:
    """시각화용 간단한 PD 제어기.

    obs = [sin(theta), cos(theta), omega] 이므로,
    atan2로 각도를 복원한 뒤 `-(kp * theta + kd * omega)` 형태의 토크를 만든다.
    이 action은 RL policy가 아니라 demo controller다.
    """
    theta = jnp.arctan2(obs[0], obs[1])
    omega = obs[2]
    torque = -(cfg.kp * theta + cfg.kd * omega)
    return jnp.array([jnp.clip(torque, -1.0, 1.0)])


jit_reset = jax.jit(reset)
jit_step = jax.jit(step)
jit_controller_action = jax.jit(controller_action, static_argnames=("cfg",))


def run_single_smoke(key: jax.Array) -> tuple[mjx.Data, jax.Array]:
    """단일 환경 smoke 실행.

    튜토리얼이 잘 붙어 있는지 제일 먼저 확인하는 가장 짧은 경로다.
    """
    data, obs = jit_reset(key)
    data.qpos.block_until_ready()

    print("initial obs:", obs)

    action = jnp.array([0.0])
    data, obs, reward, done = jit_step(data, action)
    data.qpos.block_until_ready()

    print("next obs:", obs)
    print("reward:", reward)
    print("done:", done)
    return data, obs


def run_batched_benchmark(key: jax.Array, num_envs: int, num_steps: int) -> None:
    """MJX의 장점인 batched rollout 성능을 보여준다."""
    keys = jax.random.split(key, num_envs)

    batched_reset = jax.jit(jax.vmap(reset))
    batched_step = jax.jit(jax.vmap(step))

    batched_data, _ = batched_reset(keys)
    batched_data.qpos.block_until_ready()

    actions = jnp.zeros((num_envs, 1))

    # 첫 호출은 XLA 컴파일 비용이 섞이므로 워밍업을 먼저 한 번 태운다.
    batched_data, obs, reward, done = batched_step(batched_data, actions)
    batched_data.qpos.block_until_ready()

    start = time.perf_counter()

    for _ in range(num_steps):
        batched_data, obs, reward, done = batched_step(batched_data, actions)

    batched_data.qpos.block_until_ready()
    elapsed = time.perf_counter() - start

    total_sim_steps = num_envs * num_steps
    steps_per_second = total_sim_steps / elapsed

    print("batched obs shape:", obs.shape)
    print("mean reward:", reward.mean())
    print("mean done:", done.mean())
    print("elapsed:", elapsed)
    print("environment steps/s:", steps_per_second)


def sync_viewer_state(viewer_data: mujoco.MjData, data: mjx.Data) -> None:
    """MJX device 상태를 MuJoCo host viewer 상태로 복사한다."""
    host_data = mjx.get_data(mj_model, data)
    viewer_data.qpos[:] = host_data.qpos
    viewer_data.qvel[:] = host_data.qvel
    viewer_data.act[:] = host_data.act
    viewer_data.ctrl[:] = host_data.ctrl
    viewer_data.time = host_data.time
    mujoco.mj_forward(mj_model, viewer_data)


def run_viewer_rollouts(args: argparse.Namespace) -> None:
    """viewer를 열고 여러 번 reset/rollout 하면서 막대를 눈으로 본다.

    이 함수가 바로 사용자가 원한 "시각화하면서 여러 번 해 보기" 경로다.
    viewer 창이 뜬 상태에서 에피소드가 끝나면 자동으로 reset해서 다시 시도한다.
    """
    import mujoco.viewer

    print("viewer mode: MJX state를 host viewer로 복사하면서 에피소드를 재생한다.")

    controller_cfg = ControllerCfg()
    viewer_data = mujoco.MjData(mj_model)
    key = jax.random.key(args.seed)

    with mujoco.viewer.launch_passive(
        mj_model,
        viewer_data,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        for episode_idx in range(args.episodes):
            if not viewer.is_running():
                break

            key, reset_key = jax.random.split(key)
            data, obs = jit_reset(reset_key)
            data.qpos.block_until_ready()
            sync_viewer_state(viewer_data, data)
            viewer.sync()

            episode_return = 0.0

            for step_idx in range(args.episode_steps):
                if not viewer.is_running():
                    return

                action = jit_controller_action(obs, controller_cfg)
                data, obs, reward, done = jit_step(data, action)
                data.qpos.block_until_ready()

                episode_return += float(reward)

                sync_viewer_state(viewer_data, data)
                viewer.sync()

                if args.realtime:
                    time.sleep(CTRL_DT)

                if bool(done):
                    break

            theta = float(data.qpos[0])
            omega = float(data.qvel[0])
            print(
                f"episode={episode_idx} steps={step_idx + 1} "
                f"return={episode_return:.3f} theta={theta:.3f} omega={omega:.3f}"
            )


def main() -> None:
    args = parse_args()

    print("JAX devices:", jax.devices())
    print("nq:", mj_model.nq)
    print("nv:", mj_model.nv)
    print("nu:", mj_model.nu)
    print("simulation dt:", SIM_DT)
    print("control dt:", CTRL_DT)
    print("substeps:", N_SUBSTEPS)

    key = jax.random.key(args.seed)

    # 1) 단일 환경 smoke test
    run_single_smoke(key)

    # 2) 사용자가 원하면 viewer로 여러 번 reset하면서 눈으로 보기
    if args.visualize:
        run_viewer_rollouts(args)

    # 3) MJX batched benchmark
    if not args.skip_benchmark:
        run_batched_benchmark(key, args.num_envs, args.benchmark_steps)


if __name__ == "__main__":
    main()
