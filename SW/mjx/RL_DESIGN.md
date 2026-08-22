# MJX Residual RL 설계 요약

전체 최신 명세와 실행법은 [docs/RESIDUAL_RL.md](../../docs/RESIDUAL_RL.md)를 따른다.
이 파일은 코드 옆에서 보는 짧은 architecture map이다.

```text
classical tripod nominal
  -> swing XYZ / stance Z-only residual (22-D)
  -> airborne-aware contact adaptation
  -> workspace projection
  -> analytical IK
  -> joint/rate/fixed ±8 Nm safety
```

- action contract: `cartesian_gait_residual_v2`, 22-D
- observation contract: `body_state_coarse9_touchdown6_v1`, 110-D
- body height: contact stance-foot terrain median 기준
- terrain features: heading-aligned 3×3 coarse grid + six nominal-touchdown heights
- terrain model: flat/curb/ramp/blocks/stairs/rough lane을 한 XML에 포함
- reset: lane과 1.5–4.0초 command interval을 env별로 재샘플
- level 4 dynamics: per-env friction/mass/damping/servo realization, force cap은 항상 ±8 Nm
- command PPO: `gamma=0.97`, unroll 20
- terrain PPO: `gamma=0.99`, unroll 32
- curriculum: evaluation `terrain_success` > 0.8 promote, < 0.5 demote
- flat→terrain: `--init-checkpoint`; observation semantic metadata까지 검증

필수 invariant:

```text
action=0 -> NumPy classical target == JAX nominal target (max error < 1e-4 rad)
early_landing = swing & airborne & contact
Safety > Contact > RL > Nominal
```
