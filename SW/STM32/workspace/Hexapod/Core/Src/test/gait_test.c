#include "test/gait_test.h"

#include "high_control/foot_trajectory.h"
#include "high_control/gait_manager.h"
#include "high_control/stance_trajectory.h"
#include "high_control/leg_kinematics.h"
#include "high_control/workspace_limiter.h"

#include <math.h>
#include <string.h>

/* 접촉 주기에 직전 Swing 발 위치가 유지되는지 검사한다. */
static bool GaitTest_CheckImmediateContactHold(void)
{
    FootTrajectory_Handle_t trajectory;  // 시험용 발 궤적 상태를 저장한다.
    RobotBodyTwist_t twist;               // 정지 Body Twist를 저장한다.
    RobotDroneOutput_t drone;             // 수동 보행 활성 상태를 저장한다.
    RobotGaitPhase_t gait;                 // 명시적 다리 상태를 저장한다.
    RobotEuler_t posture;                  // 수평 자세를 저장한다.
    RobotFootTargets_t before;             // 접촉 직전 발 위치를 저장한다.
    RobotFootTargets_t candidate;          // 접촉 확인 중 발 위치를 저장한다.
    RobotFootTargets_t after;              // 접촉 직후 발 위치를 저장한다.
    uint32_t leg;                          // 상태를 설정할 다리 번호를 저장한다.

    memset(&twist, 0, sizeof(twist));      // 정지 명령을 준비한다.
    memset(&drone, 0, sizeof(drone));      // 보행 출력을 초기화한다.
    memset(&gait, 0, sizeof(gait));        // 전체 Stance를 준비한다.
    memset(&posture, 0, sizeof(posture));  // 수평 자세를 준비한다.
    drone.manual_enable = true;            // 공통 Z 복구 경로를 활성화한다.
    drone.body_control_enable = true;      // 발 궤적 계산을 활성화한다.
    drone.tripod_enable = true;            // 정상 Tripod를 활성화한다.
    drone.tripod_mode = ROBOT_TRIPOD_NORMAL;  // 정상 보행 모드를 선택한다.
    FootTrajectory_Init(&trajectory);         // 기본 발 위치를 준비한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; leg += 2U)
    {
        gait.state[leg] = ROBOT_LEG_SWING;  // 1·3·5번 다리를 Swing으로 둔다.
        gait.progress[leg] = 0.75f;          // 착지 가능한 하강 구간을 선택한다.
    }
    before = FootTrajectory_Step(&trajectory, &twist, &drone,
                                 &gait, &posture);  // 접촉 직전 위치를 계산한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; leg += 2U)
    {
        if (!FootTrajectory_LatchCommandedTouchdown(&trajectory, (uint8_t)leg))
        {
            return false;
        }
        gait.state[leg] = ROBOT_LEG_TOUCHDOWN_CANDIDATE;  // 접촉 확인 대기로 전환한다.
        gait.progress[leg] = 0.755f;                      // 다음 제어 주기 진행률을 넣는다.
    }
    candidate = FootTrajectory_Step(&trajectory, &twist, &drone,
                                    &gait, &posture);  // 접촉 후보 위치를 유지한다.
    for (leg = 0U; leg < ROBOT_LEG_COUNT; leg += 2U)
    {
        gait.state[leg] = ROBOT_LEG_STANCE;  // 같은 위치에서 접촉을 확정한다.
    }
    after = FootTrajectory_Step(&trajectory, &twist, &drone,
                                &gait, &posture);  // 접촉 직후 위치를 계산한다.

    return (fabsf(candidate.foot[0].z - before.foot[0].z) < 0.0000001f) &&
           (fabsf(candidate.foot[2].z - before.foot[2].z) < 0.0000001f) &&
           (fabsf(candidate.foot[4].z - before.foot[4].z) < 0.0000001f) &&
           (fabsf(after.foot[0].z - before.foot[0].z) < 0.0000001f) &&
           (fabsf(after.foot[2].z - before.foot[2].z) < 0.0000001f) &&
           (fabsf(after.foot[4].z - before.foot[4].z) < 0.0000001f);  // 세 발 모두 추가 하강하지 않는지 확인한다.
}

/* 수동 보행 정지에서 직전 보폭과 Yaw가 발을 움직이지 않는지 검사한다. */
static bool GaitTest_CheckManualStanceHold(void)
{
    FootTrajectory_Handle_t trajectory;  // 시험용 발 궤적 상태를 저장한다.
    RobotBodyTwist_t twist;               // 남아 있는 직전 보행 명령을 저장한다.
    RobotDroneOutput_t drone;             // 수동 정지 상태를 저장한다.
    RobotGaitPhase_t gait;                 // 전체 Stance 상태를 저장한다.
    RobotEuler_t posture;                  // 수평 자세를 저장한다.
    RobotFootTargets_t before;             // 정지 전 발 위치를 저장한다.
    RobotFootTargets_t after;              // 정지 후 발 위치를 저장한다.
    uint32_t leg;                          // 비교할 다리 번호를 저장한다.
    uint32_t cycle;                        // 정지 입력과 검사 거부를 구분한다.

    memset(&twist, 0, sizeof(twist));      // 직전 보행 명령을 초기화한다.
    memset(&drone, 0, sizeof(drone));      // 수동 정지 출력을 초기화한다.
    memset(&gait, 0, sizeof(gait));        // 전체 Stance를 준비한다.
    memset(&posture, 0, sizeof(posture));  // 수평 자세를 준비한다.
    twist.vx = 0.10f;                      // 남아 있는 전진 보폭을 넣는다.
    twist.wz = 0.20f;                      // 남아 있는 Yaw 보정을 넣는다.
    drone.manual_enable = true;            // 수동 제어 상태를 활성화한다.
    drone.body_control_enable = true;      // 발 궤적 계산을 활성화한다.
    drone.tripod_mode = ROBOT_TRIPOD_NORMAL;  // 정상 보행 모드를 선택한다.
    FootTrajectory_Init(&trajectory);         // 기본 발 위치를 준비한다.
    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        before.foot[leg] = trajectory.memory[leg];  // 정지 전 발 위치를 저장한다.
        gait.state[leg] = ROBOT_LEG_STANCE;         // 여섯 발을 Stance로 둔다.
        trajectory.adapted_stance[leg] = true;     // 착지 후 실제 위치 적분을 준비한다.
    }

    for (cycle = 0U; cycle < 2U; ++cycle)
    {
        drone.tripod_enable = (cycle != 0U);  // 정지 입력과 검사 거부 후 활성 입력을 각각 넣는다.
        after = FootTrajectory_Step(&trajectory, &twist, &drone,
                                    &gait, &posture);  // 내부 보행이 꺼진 발 목표를 계산한다.
        for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
        {
            if ((fabsf(after.foot[leg].x - before.foot[leg].x) > 0.0000001f) ||
                (fabsf(after.foot[leg].y - before.foot[leg].y) > 0.0000001f) ||
                (fabsf(after.foot[leg].z - before.foot[leg].z) > 0.0000001f))
            {
                return false;  // 사용자 정지와 검사 거부 주기의 발 이동을 검출한다.
            }
        }
    }

    return true;
}

/* Tripod 공통 하강 오차가 주기 제한량만큼 복구되는지 검사한다. */
static bool GaitTest_CheckCommonZRecovery(void)
{
    FootTrajectory_Handle_t trajectory;             // 시험용 발 궤적 상태를 저장한다.
    RobotBodyTwist_t twist;                          // 정지 Body Twist를 저장한다.
    RobotDroneOutput_t drone;                        // 수동 보행 활성 상태를 저장한다.
    RobotGaitPhase_t gait;                            // 명시적 다리 상태를 저장한다.
    RobotEuler_t posture;                             // 수평 자세를 저장한다.
    RobotVec3_t measured_foot[ROBOT_LEG_COUNT];      // PWM FK에 대응할 발 위치를 저장한다.
    float recovery_before;                           // Swing 전 복구 잔량을 저장한다.
    uint32_t leg;                                    // 상태를 설정할 다리 번호를 저장한다.

    memset(&twist, 0, sizeof(twist));      // 정지 명령을 준비한다.
    memset(&drone, 0, sizeof(drone));      // 보행 출력을 초기화한다.
    memset(&gait, 0, sizeof(gait));        // 전체 Stance를 준비한다.
    memset(&posture, 0, sizeof(posture));  // 수평 자세를 준비한다.
    drone.manual_enable = true;            // 공통 Z 복구 경로를 활성화한다.
    drone.body_control_enable = true;      // 발 궤적 계산을 활성화한다.
    drone.tripod_enable = true;            // 정상 Tripod를 활성화한다.
    drone.tripod_mode = ROBOT_TRIPOD_NORMAL;  // 정상 보행 모드를 선택한다.
    gait.enabled_internal = true;             // 실제 보행 중 지형 Z 적용을 활성화한다.
    FootTrajectory_Init(&trajectory);         // 기본 발 위치를 준비한다.
    LegKinematics_GetBaseFeet(measured_foot); // 기본 PWM 발 위치를 준비한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if ((leg % 2U) == 0U)
        {
            measured_foot[leg].z -= 0.003f;  // 1·3·5번 PWM 발에 공통 -3 mm 오차를 넣는다.
        }
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; leg += 2U)
    {
        if (!FootTrajectory_LatchTouchdown(&trajectory, (uint8_t)leg,
                                           &measured_foot[leg],
                                           &posture))
        {
            return false;
        }
        gait.state[leg] = ROBOT_LEG_STANCE;  // 1·3·5번을 접촉 Stance로 둔다.
        gait.progress[leg] = 0.75f;           // 조기 착지 진행률을 넣는다.
    }
    FootTrajectory_UpdateCommandedLanding(&trajectory, &gait, true);  // 접촉 확인 후 PWM 명령 오차를 수집한다.
    if (fabsf(trajectory.common_z_recovery_remaining[0] - 0.003f) > 0.000001f)
    {
        return false;
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; leg += 2U)
    {
        gait.state[leg] = ROBOT_LEG_SWING;  // 해당 Tripod를 Swing으로 전환한다.
    }
    recovery_before = trajectory.common_z_recovery_remaining[0];  // Swing 전 복구 잔량을 저장한다.
    (void)FootTrajectory_Step(&trajectory, &twist, &drone,
                              &gait, &posture);  // Swing 중 복구를 보류한다.
    if (fabsf(trajectory.common_z_recovery_remaining[0] -
              recovery_before) > 0.0000001f)
    {
        return false;
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; leg += 2U)
    {
        gait.state[leg] = ROBOT_LEG_STANCE;  // 해당 Tripod를 다시 Stance로 전환한다.
    }
    (void)FootTrajectory_Step(&trajectory, &twist, &drone,
                              &gait, &posture);  // 남은 공통 Z 복구를 재개한다.
    {
        const float progress = ROBOT_CONTROL_PERIOD_S /
                               ROBOT_COMMON_Z_RECOVERY_TIME_S;  // 첫 S-curve 진행률을 계산한다.
        const float expected_step = recovery_before * progress * progress *
                                    (3.0f - 2.0f * progress);  // 첫 Smoothstep 복구량을 계산한다.

        return fabsf((recovery_before -
                      trajectory.common_z_recovery_remaining[0]) -
                     expected_step) < 0.0000001f;  // Stance 복귀 후 느린 시작량을 확인한다.
    }
}

/* Late Landing이 100 mm에서 전원을 유지한 채 멈추는지 검사한다. */
static bool GaitTest_CheckLateLandingLimit(void)
{
    FootTrajectory_Handle_t trajectory;       // 시험용 발 궤적 상태를 저장한다.
    GaitManager_Handle_t manager;              // 시험용 Late Landing 시간을 저장한다.
    LegKinematics_Handle_t kinematics;         // 최종 제한 IK 상태를 저장한다.
    RobotBodyTwist_t twist;                    // 정지 Body Twist를 저장한다.
    RobotDroneOutput_t drone;                  // 전체 발 착지 출력을 저장한다.
    RobotGaitPhase_t gait;                     // Late Landing 상태를 저장한다.
    RobotFootTargets_t feet;                   // Late Landing 발 위치를 저장한다.
    RobotFootTargets_t held;                   // 한계 도달 후 발 위치를 저장한다.
    RobotVec3_t base[ROBOT_LEG_COUNT];          // 탐색 전 기본 발 위치를 저장한다.
    RobotEuler_t posture;                      // 수평 자세를 저장한다.
    bool contact[ROBOT_LEG_COUNT];             // 미접촉 발 상태를 저장한다.
    uint32_t cycle;                            // 탐색 제어 주기를 저장한다.
    uint32_t leg;                              // 검사할 다리 번호를 저장한다.

    memset(&twist, 0, sizeof(twist));          // 정지 명령을 준비한다.
    memset(&drone, 0, sizeof(drone));          // 착지 출력을 초기화한다.
    memset(&posture, 0, sizeof(posture));      // 수평 자세를 준비한다.
    memset(contact, 0, sizeof(contact));       // 모든 발을 미접촉으로 둔다.
    drone.tripod_enable = true;                // 전체 발 착지 궤적을 활성화한다.
    drone.tripod_mode = ROBOT_TRIPOD_LAND_ALL; // 전체 발 Late Landing을 선택한다.
    FootTrajectory_Init(&trajectory);          // 기본 발 위치를 준비한다.
    GaitManager_Init(&manager);                // 탐색 시간을 초기화한다.
    LegKinematics_Init(&kinematics);            // 최종 IK 상태를 초기화한다.
    LegKinematics_GetBaseFeet(base);             // 탐색 거리 기준 위치를 읽는다.

    for (cycle = 0U;
         cycle <= ((uint32_t)(ROBOT_LATE_LANDING_MAX_TIME_S /
                              ROBOT_CONTROL_PERIOD_S) + 1U);
         ++cycle)
    {
        gait = GaitManager_Step(&manager, false, true, true, true,
                                ROBOT_TRIPOD_LAND_ALL,
                                0.0f, contact);  // 미접촉 탐색 시간을 진행한다.
        feet = FootTrajectory_Step(&trajectory, &twist, &drone,
                                   &gait, &posture);  // 제한 전까지 발을 내린다.
        if (gait.late_landing_stop)
        {
            break;  // 100 mm 탐색 완료 정지에서 반복을 끝낸다.
        }
    }

    if (!gait.late_landing_stop || !gait.late_landing_exhausted[0] ||
        !gait.late_landing_hold || !manager.late_landing_hold ||
        (fabsf((base[0].z - feet.foot[0].z) -
               ROBOT_LATE_LANDING_MAX_DISTANCE_M) > 0.000001f))
    {
        return false;
    }

    gait = GaitManager_Step(&manager, false, true, true, true,
                            ROBOT_TRIPOD_LAND_ALL,
                            0.0f, contact);  // 한계 상태를 한 주기 더 유지한다.
    held = FootTrajectory_Step(&trajectory, &twist, &drone,
                               &gait, &posture);  // 한계 이후 좌표를 유지한다.
    if (!gait.waiting_start || !gait.late_landing_hold ||
        gait.enabled_internal ||
        (fabsf(held.foot[0].z - feet.foot[0].z) > 0.0000001f))
    {
        return false;
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        RobotVec3_t limited;                // 최종 제한 발 위치를 저장한다.
        float angle[ROBOT_JOINTS_PER_LEG];  // 최종 IK 해를 저장한다.
        bool was_limited;                   // 최종 제한 여부를 저장한다.

        if (!LegKinematics_LimitFoot((uint8_t)leg, &held.foot[leg],
                                     &limited, &was_limited) ||
            !LegKinematics_Inverse(&kinematics, (uint8_t)leg,
                                   &limited, angle))
        {
            return false;
        }
    }

    return true;
}

/* 첫 보행이 0.1초 대기 후 세 지점 검사를 요청하는지 확인한다. */
static bool GaitTest_CheckStartupPreview(void)
{
    GaitManager_Handle_t manager;   // 시험용 첫 보행 상태를 저장한다.
    RobotGaitPhase_t gait;          // 첫 보행 다리 상태를 저장한다.
    bool contact[ROBOT_LEG_COUNT];  // 시험용 발 접촉 상태를 저장한다.
    uint32_t cycle;                 // 입력 안정 대기 주기를 저장한다.

    memset(contact, 0, sizeof(contact));  // Swing 허용용 미접촉 상태를 만든다.
    GaitManager_Init(&manager);           // 첫 보행 상태를 초기화한다.

    for (cycle = 0U; cycle < ROBOT_GAIT_START_DELAY_CYCLES; ++cycle)
    {
        gait = GaitManager_Step(&manager, true, true, false, false,
                                ROBOT_TRIPOD_NORMAL,
                                0.0f, contact);  // 0.1초 동안 입력을 안정시킨다.
        if (!gait.enabled_internal || !gait.startup_phase ||
            !gait.waiting_start || gait.next_phase_preview)
        {
            return false;
        }
    }

    gait = GaitManager_Step(&manager, true, true, false, false,
                            ROBOT_TRIPOD_NORMAL,
                            0.0f, contact);  // 대기 후 세 지점 검사를 요청한다.
    if (!gait.enabled_internal || !gait.startup_phase || !gait.waiting_start ||
        !gait.next_phase_preview || (gait.next_phase_swing_mask != 0x15U))
    {
        return false;
    }

    gait = GaitManager_Step(&manager, true, true, true, true,
                            ROBOT_TRIPOD_NORMAL,
                            0.0f, contact);  // 검사 통과 후 첫 위상을 시작한다.
    return (gait.state[0] == ROBOT_LEG_SWING) &&
           (gait.state[1] == ROBOT_LEG_STANCE) &&
           (gait.progress[0] == 0.0f);  // 1·3·5 Swing이 진행률 0에서 시작하는지 확인한다.
}

/* 검증을 마친 속도가 각 보행 위상 동안 고정되는지 검사한다. */
static bool GaitTest_CheckPhaseTwistLatch(void)
{
    FootTrajectory_Handle_t trajectory;  // 시험용 위상 속도 기억을 저장한다.
    RobotBodyTwist_t twist;               // 시간에 따라 바꿀 속도를 저장한다.
    RobotDroneOutput_t drone;             // 수동 보행 활성 상태를 저장한다.
    RobotGaitPhase_t gait;                 // 시험용 Tripod 위상을 저장한다.
    RobotFootTargets_t phase_end;          // 첫 위상 마지막 발 위치를 저장한다.
    RobotFootTargets_t phase_start;        // 다음 위상 첫 발 위치를 저장한다.
    RobotEuler_t posture;                  // 수평 자세를 저장한다.
    uint32_t leg;                          // Swing 그룹을 설정할 다리 번호를 저장한다.

    memset(&twist, 0, sizeof(twist));      // 첫 속도 명령을 초기화한다.
    memset(&drone, 0, sizeof(drone));      // 수동 보행 출력을 초기화한다.
    memset(&gait, 0, sizeof(gait));        // 전체 Stance 상태를 준비한다.
    memset(&posture, 0, sizeof(posture));  // 수평 자세를 준비한다.
    drone.manual_enable = true;            // 위상 속도 고정을 활성화한다.
    drone.body_control_enable = true;      // 발 궤적 계산을 활성화한다.
    drone.tripod_enable = true;            // 정상 Tripod를 활성화한다.
    drone.tripod_mode = ROBOT_TRIPOD_NORMAL;  // 정상 보행 모드를 선택한다.
    gait.enabled_internal = true;             // 내부 보행 상태를 활성화한다.
    gait.startup_phase = true;                // 첫 위상을 선택한다.
    for (leg = 0U; leg < ROBOT_LEG_COUNT; leg += 2U)
    {
        gait.state[leg] = ROBOT_LEG_SWING;  // 1·3·5번 다리를 첫 Swing으로 둔다.
    }
    FootTrajectory_Init(&trajectory);  // 발 위치와 속도 이력을 초기화한다.

    twist.vx = 0.10f;  // 첫 세 지점 검사 뒤 속도를 넣는다.
    (void)FootTrajectory_Step(&trajectory, &twist, &drone,
                              &gait, &posture);  // 첫 위상 속도를 고정한다.
    twist.vx = 0.0f;   // 첫 위상 중 정지 입력을 넣는다.
    (void)FootTrajectory_Step(&trajectory, &twist, &drone,
                              &gait, &posture);  // 중간 입력 변경을 적용한다.
    if (fabsf(trajectory.phase_twist.vx - 0.10f) > 0.0000001f)
    {
        return false;  // 첫 위상 보폭이 중간 입력으로 바뀌지 않는지 확인한다.
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        gait.progress[leg] = 1.0f;  // 첫 위상을 끝점까지 진행한다.
    }
    phase_end = FootTrajectory_Step(&trajectory, &twist, &drone,
                                    &gait, &posture);  // 기존 0.10 m/s 보폭을 완료한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        gait.state[leg] = ((leg % 2U) == 0U)
                        ? ROBOT_LEG_STANCE
                        : ROBOT_LEG_SWING;  // 2·4·6번 다리를 다음 Swing으로 바꾼다.
        gait.progress[leg] = 0.0f;          // 다음 위상을 시작점에 둔다.
    }
    gait.startup_phase = false;  // 반복 보행 위상을 선택한다.
    twist.vx = 0.20f;            // 착륙 시 검증을 마친 다음 속도를 넣는다.
    phase_start = FootTrajectory_Step(&trajectory, &twist, &drone,
                                      &gait, &posture);  // 검증한 속도를 다음 위상에 고정한다.
    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if ((fabsf(phase_start.foot[leg].x - phase_end.foot[leg].x) > 0.0000001f) ||
            (fabsf(phase_start.foot[leg].y - phase_end.foot[leg].y) > 0.0000001f) ||
            (fabsf(phase_start.foot[leg].z - phase_end.foot[leg].z) > 0.0000001f))
        {
            return false;  // 속도 변경 위상 경계에서 발 목표가 튀지 않는지 확인한다.
        }
    }
    twist.vx = 0.60f;  // 다음 위상 중 바뀐 입력을 넣는다.
    (void)FootTrajectory_Step(&trajectory, &twist, &drone,
                              &gait, &posture);  // 현재 위상의 고정 속도를 유지한다.
    return fabsf(trajectory.phase_twist.vx - 0.20f) < 0.0000001f;  // 0.20 m/s 고정을 확인한다.
}

/* 입력 안정 대기와 세 지점 검사를 통과시켜 정상 보행을 시작한다. */
static RobotGaitPhase_t GaitTest_StartNormal(GaitManager_Handle_t *manager,
                                             const bool contact[ROBOT_LEG_COUNT])
{
    uint32_t cycle;  // 입력 안정 대기 주기를 저장한다.

    for (cycle = 0U; cycle < ROBOT_GAIT_START_DELAY_CYCLES; ++cycle)
    {
        (void)GaitManager_Step(manager, true, true, false, false,
                               ROBOT_TRIPOD_NORMAL,
                               0.0f, contact);  // 첫 입력 안정 대기를 진행한다.
    }
    (void)GaitManager_Step(manager, true, true, false, false,
                           ROBOT_TRIPOD_NORMAL,
                           0.0f, contact);  // 첫 위상 세 지점 검사를 요청한다.

    return GaitManager_Step(manager, true, true, true, true,
                            ROBOT_TRIPOD_NORMAL,
                            0.0f, contact);  // 검사 통과 결과로 첫 위상을 시작한다.
}

/* 조기 착지 뒤에도 위상 끝에서 검사와 다음 전환을 수행하는지 확인한다. */
static bool GaitTest_CheckPreviewTransition(void)
{
    GaitManager_Handle_t manager;   // 시험용 다음 위상 상태를 저장한다.
    RobotGaitPhase_t gait;          // 다음 위상 검사 신호를 저장한다.
    bool contact[ROBOT_LEG_COUNT];  // Swing 비접촉과 착지를 저장한다.
    uint32_t cycle;                 // 진행할 제어 주기를 저장한다.

    memset(contact, 0, sizeof(contact));  // 첫 Swing의 비접촉 상태를 만든다.
    GaitManager_Init(&manager);           // 최초 보행 상태를 초기화한다.
    gait = GaitTest_StartNormal(&manager, contact);  // 첫 1·3·5 Swing을 시작한다.

    for (cycle = 0U;
         cycle < (uint32_t)(ROBOT_EARLY_LANDING_PROGRESS *
                            ROBOT_GAIT_PHASE_TIME_S /
                            ROBOT_CONTROL_PERIOD_S);
         ++cycle)
    {
        gait = GaitManager_Step(&manager, true, true, false, false,
                                ROBOT_TRIPOD_NORMAL,
                                0.0f, contact);  // Early Landing 시점까지 Swing을 진행한다.
    }

    memset(contact, 1, sizeof(contact));  // 여섯 발의 확정 착지를 만든다.
    gait = GaitManager_Step(&manager, true, true, false, false,
                            ROBOT_TRIPOD_NORMAL,
                            0.0f, contact);  // 조기 착지를 확정하되 현재 위상을 유지한다.
    if (gait.next_phase_preview || (manager.phase_index != 0U) ||
        (manager.phase_time_s >= ROBOT_GAIT_PHASE_TIME_S))
    {
        return false;
    }

    while (manager.phase_cycle_count <
           ((uint32_t)(ROBOT_GAIT_PHASE_TIME_S / ROBOT_CONTROL_PERIOD_S) - 1U))
    {
        gait = GaitManager_Step(&manager, true, true, false, false,
                                ROBOT_TRIPOD_NORMAL,
                                0.0f, contact);  // 착지한 발로 남은 위상을 완료한다.
        if (gait.next_phase_preview || (manager.phase_index != 0U))
        {
            return false;
        }
    }

    gait = GaitManager_Step(&manager, true, true, false, false,
                            ROBOT_TRIPOD_NORMAL,
                            0.0f, contact);  // 실제 위상 끝에서 세 지점 검사를 요청한다.
    if (!gait.next_phase_preview || (gait.next_phase_swing_mask != 0x2AU) ||
        !gait.waiting_start || !manager.next_phase_enable ||
        !manager.next_phase_locked ||
        (manager.phase_time_s < ROBOT_GAIT_PHASE_TIME_S))
    {
        return false;
    }

    gait = GaitManager_Step(&manager, true, true, true, true,
                            ROBOT_TRIPOD_NORMAL,
                            0.0f, contact);  // 세 지점 통과 뒤 다음 위상으로 전환한다.
    return (manager.phase_index == 1U) &&
           (gait.state[0] == ROBOT_LEG_STANCE) &&
           (gait.state[1] == ROBOT_LEG_SWING) &&
           (gait.progress[1] == 0.0f);  // 2·4·6 Swing의 즉시 시작을 확인한다.
}

/* 한 번의 보행 입력이 두 Tripod 걸음을 완료하는지 검사한다. */
static bool GaitTest_CheckTwoStepCommandLatch(void)
{
    GaitManager_Handle_t manager;            // 시험용 두 걸음 Latch 상태를 저장한다.
    RobotGaitPhase_t gait;                   // 현재 Tripod 다리 상태를 저장한다.
    bool contact[ROBOT_LEG_COUNT];           // 여섯 발의 확정 접촉을 저장한다.
    uint32_t guard;                          // 비정상 무한 진행을 차단한다.

    memset(contact, 1, sizeof(contact));  // 두 걸음의 확정 착지를 준비한다.
    GaitManager_Init(&manager);           // 최초 보행 상태를 초기화한다.
    gait = GaitManager_Step(&manager, true, true, false, false,
                            ROBOT_TRIPOD_NORMAL,
                            0.0f, contact);  // 한 제어 주기만 보행 입력을 넣는다.

    guard = 0U;  // 첫 위상 검사 대기 한계를 초기화한다.
    while (!gait.next_phase_preview && (guard < 1000U))
    {
        gait = GaitManager_Step(&manager, true, false, false, false,
                                ROBOT_TRIPOD_NORMAL,
                                0.0f, contact);  // 입력 해제 후에도 첫 위상 검사를 기다린다.
        guard++;  // 대기 주기를 누적한다.
    }
    if (!gait.next_phase_preview || !manager.run_enable)
    {
        return false;  // 짧은 시작 입력이 첫 걸음까지 유지되는지 확인한다.
    }

    gait = GaitManager_Step(&manager, true, false, true, true,
                            ROBOT_TRIPOD_NORMAL,
                            0.0f, contact);  // 첫 위상 검사를 통과시킨다.
    guard = 0U;  // 첫 걸음 완료 대기 한계를 초기화한다.
    while (!gait.next_phase_preview && (guard < 1000U))
    {
        gait = GaitManager_Step(&manager, true, false, false, false,
                                ROBOT_TRIPOD_NORMAL,
                                0.0f, contact);  // 정지 입력 상태로 첫 걸음을 완료한다.
        guard++;  // 첫 걸음 주기를 누적한다.
    }
    if (!gait.next_phase_preview ||
        (gait.next_phase_swing_mask != 0x2AU))
    {
        return false;  // 첫 걸음 뒤 반대 Tripod 검사가 강제되는지 확인한다.
    }

    gait = GaitManager_Step(&manager, true, false, true, true,
                            ROBOT_TRIPOD_NORMAL,
                            0.0f, contact);  // 같은 속도의 둘째 걸음을 시작한다.
    if (!gait.enabled_internal || (manager.phase_index != 1U) ||
        (manager.command_pair_step_count != 1U))
    {
        return false;
    }

    guard = 0U;  // 둘째 걸음 완료 대기 한계를 초기화한다.
    while (gait.enabled_internal && (guard < 1000U))
    {
        gait = GaitManager_Step(&manager, true, false, false, false,
                                ROBOT_TRIPOD_NORMAL,
                                0.0f, contact);  // 정지 입력을 둘째 걸음 끝에서 반영한다.
        guard++;  // 둘째 걸음 주기를 누적한다.
    }
    return !gait.enabled_internal && (manager.phase_index == 2U) &&
           (manager.command_pair_step_count == 0U);  // 정확히 두 걸음 뒤 정지하는지 확인한다.
}

/* 실제 접촉 후보와 확정 입력으로 첫 정상 위상을 시작한다. */
static RobotGaitPhase_t GaitTest_StartConfirmedNormal(
    GaitManager_Handle_t *manager,
    bool contact[ROBOT_LEG_COUNT],
    bool contact_raw[ROBOT_LEG_COUNT])
{
    uint32_t cycle;  // 입력 안정 대기 주기를 저장한다.

    memset(contact, 1, sizeof(bool) * ROBOT_LEG_COUNT);      // 시작 전 여섯 발 접촉을 확정한다.
    memset(contact_raw, 1, sizeof(bool) * ROBOT_LEG_COUNT);  // 시작 전 여섯 접촉 후보를 준비한다.
    for (cycle = 0U; cycle < ROBOT_GAIT_START_DELAY_CYCLES; ++cycle)
    {
        (void)GaitManager_StepContacts(manager, true, true, false, false,
                                       ROBOT_TRIPOD_NORMAL, 0.0f,
                                       contact, contact_raw);  // 첫 입력 안정 대기를 진행한다.
    }
    (void)GaitManager_StepContacts(manager, true, true, false, false,
                                   ROBOT_TRIPOD_NORMAL, 0.0f,
                                   contact, contact_raw);  // 첫 위상 세 지점 검사를 요청한다.
    return GaitManager_StepContacts(manager, true, true, true, true,
                                    ROBOT_TRIPOD_NORMAL, 0.0f,
                                    contact, contact_raw);  // 검사 통과로 첫 위상을 시작한다.
}

/* 접촉 후보 취소 뒤 남은 Swing을 이어가는지 검사한다. */
static bool GaitTest_CheckTouchdownConfirmation(void)
{
    GaitManager_Handle_t manager;             // 접촉 확인 시험 상태를 저장한다.
    RobotGaitPhase_t gait;                    // 접촉 후보 다리 상태를 저장한다.
    bool contact[ROBOT_LEG_COUNT];            // 확정 접촉을 저장한다.
    bool contact_raw[ROBOT_LEG_COUNT];        // 접촉 후보를 저장한다.
    const uint32_t early_cycles =
        (uint32_t)((ROBOT_GAIT_PHASE_TIME_S * ROBOT_EARLY_LANDING_PROGRESS) /
                   ROBOT_CONTROL_PERIOD_S) + 1U;  // 조기 착지 허용 직후 주기를 계산한다.
    uint32_t cycle;                           // 하강 구간까지 진행할 주기를 저장한다.

    GaitManager_Init(&manager);  // 첫 정상 위상을 준비한다.
    gait = GaitTest_StartConfirmedNormal(&manager, contact, contact_raw);  // 접촉 확인 시험을 시작한다.
    contact[0] = false; contact[2] = false; contact[4] = false;          // Swing 그룹 비접촉을 확정한다.
    contact_raw[0] = false; contact_raw[2] = false; contact_raw[4] = false;  // Swing 접촉 후보를 제거한다.

    for (cycle = 0U; cycle < early_cycles; ++cycle)
    {
        gait = GaitManager_StepContacts(&manager, true, true, false, false,
                                        ROBOT_TRIPOD_NORMAL, 0.0f,
                                        contact, contact_raw);  // 착지 허용 하강 구간까지 진행한다.
    }
    contact_raw[0] = true;  // 1번 발의 짧은 접촉 후보를 넣는다.
    gait = GaitManager_StepContacts(&manager, true, true, false, false,
                                    ROBOT_TRIPOD_NORMAL, 0.0f,
                                    contact, contact_raw);  // 확정 전 발을 일시정지한다.
    if ((gait.state[0] != ROBOT_LEG_TOUCHDOWN_CANDIDATE) || manager.landed[0])
    {
        return false;
    }

    contact_raw[0] = false;  // 10 ms 전에 접촉 후보를 취소한다.
    gait = GaitManager_StepContacts(&manager, true, true, false, false,
                                    ROBOT_TRIPOD_NORMAL, 0.0f,
                                    contact, contact_raw);  // 고정 위치에서 남은 Swing을 재개한다.
    if (gait.state[0] != ROBOT_LEG_SWING)
    {
        return false;
    }

    contact_raw[0] = true;  // 다시 접촉 후보를 넣는다.
    contact[0] = true;      // 10 ms 연속 확인 완료를 전달한다.
    gait = GaitManager_StepContacts(&manager, true, true, false, false,
                                    ROBOT_TRIPOD_NORMAL, 0.0f,
                                    contact, contact_raw);  // 확정 접촉을 Stance로 전환한다.
    return manager.landed[0] && (gait.state[0] == ROBOT_LEG_STANCE);
}

/* 지지발 이탈에서 위상을 멈추고 재접촉 뒤 같은 걸음을 재개하는지 검사한다. */
static bool GaitTest_CheckSupportRecovery(void)
{
    GaitManager_Handle_t manager;       // 지지발 재착지 상태를 저장한다.
    RobotGaitPhase_t gait;              // 일시정지 다리 상태를 저장한다.
    bool contact[ROBOT_LEG_COUNT];      // 확정 접촉을 저장한다.
    bool contact_raw[ROBOT_LEG_COUNT];  // 접촉 후보를 저장한다.
    uint32_t frozen_cycle;              // 정지 순간 위상 주기를 저장한다.
    uint32_t cycle;                     // 시험 보행 주기를 저장한다.

    GaitManager_Init(&manager);  // 첫 정상 위상을 준비한다.
    gait = GaitTest_StartConfirmedNormal(&manager, contact, contact_raw);  // 지지발 시험을 시작한다.
    contact[0] = false; contact[2] = false; contact[4] = false;          // Swing 그룹 비접촉을 확정한다.
    contact_raw[0] = false; contact_raw[2] = false; contact_raw[4] = false;  // Swing 접촉 후보를 제거한다.
    for (cycle = 0U; cycle < 20U; ++cycle)
    {
        gait = GaitManager_StepContacts(&manager, true, true, false, false,
                                        ROBOT_TRIPOD_NORMAL, 0.0f,
                                        contact, contact_raw);  // 보행 중간까지 진행한다.
    }

    contact[1] = false;      // 기존 2번 지지발의 해제를 확정한다.
    contact_raw[1] = false;  // 2번 지지발 접촉 후보를 제거한다.
    frozen_cycle = manager.phase_cycle_count;  // 정지 전 위상 주기를 저장한다.
    gait = GaitManager_StepContacts(&manager, true, true, false, false,
                                    ROBOT_TRIPOD_NORMAL, 0.0f,
                                    contact, contact_raw);  // 지지발 재착지를 시작한다.
    if (!gait.support_recovery_active ||
        (gait.support_recovery_mask != 0x2AU) ||
        (gait.state[1] != ROBOT_LEG_LATE_LANDING) ||
        (manager.phase_cycle_count != frozen_cycle))
    {
        return false;
    }

    contact_raw[1] = true;  // 2번 지지발의 재접촉 후보를 넣는다.
    gait = GaitManager_StepContacts(&manager, true, true, false, false,
                                    ROBOT_TRIPOD_NORMAL, 0.0f,
                                    contact, contact_raw);  // 접촉 확인 중 하강을 멈춘다.
    if ((gait.state[1] != ROBOT_LEG_TOUCHDOWN_CANDIDATE) ||
        (manager.phase_cycle_count != frozen_cycle))
    {
        return false;
    }

    contact[1] = true;  // 2번 지지발의 10 ms 접촉 확정을 넣는다.
    gait = GaitManager_StepContacts(&manager, true, true, false, false,
                                    ROBOT_TRIPOD_NORMAL, 0.0f,
                                    contact, contact_raw);  // 기존 위상 진행을 재개한다.
    return !gait.support_recovery_active &&
           (manager.phase_cycle_count == (frozen_cycle + 1U)) &&
           (gait.state[0] == ROBOT_LEG_SWING) &&
           (gait.state[1] == ROBOT_LEG_STANCE);  // 같은 Swing·Stance 역할로 재개되는지 확인한다.
}

/* 조기 착지발의 반복 이탈 뒤 공중발이 기존 Swing 경로를 이어가는지 검사한다. */
static bool GaitTest_CheckSupportRecoveryTrajectory(bool resume_before_hold,
                                                    bool candidate_before_hold)
{
    GaitManager_Handle_t manager[2];        // 기준 보행과 일시정지 보행 상태를 저장한다.
    FootTrajectory_Handle_t trajectory[2];  // 두 보행의 독립 궤적 상태를 저장한다.
    RobotGaitPhase_t gait[2];               // 두 보행의 현재 위상을 저장한다.
    RobotFootTargets_t targets[2];          // 같은 위상에서 비교할 발 위치를 저장한다.
    RobotBodyTwist_t twist = {0};           // 고정 보폭과 회전 명령을 저장한다.
    RobotDroneOutput_t drone = {0};         // 정상 수동 보행 명령을 저장한다.
    RobotEuler_t posture = {0};             // 수평 몸체 자세를 저장한다.
    bool contact[2][ROBOT_LEG_COUNT];       // 두 보행의 확정 접촉을 저장한다.
    bool contact_raw[2][ROBOT_LEG_COUNT];   // 두 보행의 접촉 후보를 저장한다.
    uint32_t run;                           // 기준 보행과 일시정지 보행을 선택한다.
    uint32_t cycle;                         // 정지 시간을 제외한 위상 주기를 저장한다.
    uint32_t leg;                           // 비교할 공중발 번호를 저장한다.
    const uint32_t phase_cycles =
        (uint32_t)(ROBOT_GAIT_PHASE_TIME_S / ROBOT_CONTROL_PERIOD_S);  // 한 위상의 전체 주기를 계산한다.
    const uint32_t candidate_cycle = (phase_cycles * 55U) / 100U;      // 기존 접촉 취소 매핑을 만들 주기를 정한다.
    const uint32_t landing_cycle = (phase_cycles * 65U) / 100U;        // 1번 발의 조기 착지 주기를 정한다.
    const uint32_t first_hold_cycle = (phase_cycles * 82U) / 100U;     // 남은 공중발의 하강 중 첫 정지를 정한다.
    const uint32_t second_hold_cycle = (phase_cycles * 91U) / 100U;    // 같은 하강 경로의 반복 정지를 정한다.

    twist.vx = 0.04f;                         // X 경로 재생성을 검출할 전진 속도를 넣는다.
    twist.vy = 0.02f;                         // Y 경로 재생성을 검출할 횡이동 속도를 넣는다.
    twist.wz = 0.10f;                         // 회전 중 발 경로를 함께 검사한다.
    drone.manual_enable = true;               // 위상 속도 고정을 활성화한다.
    drone.body_control_enable = true;         // 발 궤적 계산을 활성화한다.
    drone.tripod_enable = true;               // 정상 Tripod 보행을 활성화한다.
    drone.tripod_mode = ROBOT_TRIPOD_NORMAL;  // 일반 재접촉 경로를 선택한다.

    for (run = 0U; run < 2U; ++run)
    {
        GaitManager_Init(&manager[run]);        // 독립된 첫 위상을 준비한다.
        FootTrajectory_Init(&trajectory[run]);  // 독립된 궤적 기억을 준비한다.
        gait[run] = GaitTest_StartConfirmedNormal(&manager[run], contact[run],
                                                  contact_raw[run]);  // 같은 접촉 입력으로 첫 위상을 시작한다.
        targets[run] = FootTrajectory_Step(&trajectory[run], &twist, &drone,
                                            &gait[run], &posture);  // 위상 시작점에서 Swing 기억을 만든다.
        for (leg = 0U; leg < ROBOT_LEG_COUNT; leg += 2U)
        {
            contact[run][leg] = false;      // Swing 그룹의 이륙을 확정한다.
            contact_raw[run][leg] = false;  // Swing 그룹의 접촉 후보를 제거한다.
        }
    }

    for (cycle = 1U; cycle <= phase_cycles; ++cycle)
    {
        const RobotFootTargets_t previous = targets[1];  // 추가 상승과 HOLD 이동을 검사할 직전 위치를 보존한다.

        for (run = 0U; run < 2U; ++run)
        {
            contact[run][0] = (cycle >= landing_cycle);  // 1번 발만 먼저 착지시킨다.
            contact_raw[run][0] = contact[run][0];       // 조기 착지의 접촉 후보를 함께 넣는다.
            contact_raw[run][2] =
                (resume_before_hold && (cycle == candidate_cycle)) ||
                (candidate_before_hold && (cycle == (first_hold_cycle - 1U)));  // 3번 발의 접촉 취소 시점을 선택한다.
        }

        if ((cycle == first_hold_cycle) || (cycle == second_hold_cycle))
        {
            const uint32_t frozen_cycle = manager[1].phase_cycle_count;  // 일시정지 직전 위상을 보존한다.
            uint32_t hold_cycle;                                         // 확정 재접촉까지의 정지 주기를 저장한다.

            if (!manager[1].landed[0])
            {
                return false;  // 조기 착지 확정 없이 재접촉 시험에 진입하는 오류를 검출한다.
            }
            contact[1][0] = false;  // 조기 착지한 1번 발을 다시 이탈시킨다.
            for (hold_cycle = 0U; hold_cycle < 4U; ++hold_cycle)
            {
                contact_raw[1][0] = (hold_cycle == 3U);  // 마지막 정지 주기에 재접촉 후보만 넣는다.
                gait[1] = GaitManager_StepContacts(&manager[1], true, true, false, false,
                                                    ROBOT_TRIPOD_NORMAL, 0.0f,
                                                    contact[1], contact_raw[1]);  // 한쪽 보행에만 지지 회복 대기를 삽입한다.
                targets[1] = FootTrajectory_Step(&trajectory[1], &twist, &drone,
                                                  &gait[1], &posture);  // 정지 위상에서 실제 발 궤적을 계산한다.
                if (!gait[1].support_recovery_active ||
                    (manager[1].phase_cycle_count != frozen_cycle) ||
                    (gait[1].state[0] != ((hold_cycle == 3U) ?
                        ROBOT_LEG_TOUCHDOWN_CANDIDATE : ROBOT_LEG_LATE_LANDING)))
                {
                    return false;  // 조기 착지발 이탈에서 재접촉 위상이 멈추지 않는 오류를 검출한다.
                }
                for (leg = 2U; leg < ROBOT_LEG_COUNT; leg += 2U)
                {
                    if ((gait[1].state[leg] != ROBOT_LEG_HOLD) ||
                        (fabsf(targets[1].foot[leg].x - previous.foot[leg].x) > 1.0e-6f) ||
                        (fabsf(targets[1].foot[leg].y - previous.foot[leg].y) > 1.0e-6f) ||
                        (fabsf(targets[1].foot[leg].z - previous.foot[leg].z) > 1.0e-6f))
                    {
                        return false;  // 재접촉 대기 중 다른 공중발이 움직이는 오류를 검출한다.
                    }
                }
            }
            contact[1][0] = true;      // 조기 착지발의 재접촉을 확정한다.
            contact_raw[1][0] = true;  // 재접촉 후보를 유지한 채 기존 위상을 재개한다.
        }

        for (run = 0U; run < 2U; ++run)
        {
            gait[run] = GaitManager_StepContacts(&manager[run], true, true, false, false,
                                                 ROBOT_TRIPOD_NORMAL, 0.0f,
                                                 contact[run], contact_raw[run]);  // 두 보행을 같은 다음 위상으로 진행한다.
            targets[run] = FootTrajectory_Step(&trajectory[run], &twist, &drone,
                                                &gait[run], &posture);  // 정지 이력만 다른 두 궤적을 계산한다.
        }
        if (gait[1].support_recovery_active ||
            (manager[0].phase_cycle_count != manager[1].phase_cycle_count))
        {
            return false;  // 재접촉 후 위상이 기준 보행과 달라지는 오류를 검출한다.
        }
        for (leg = 2U; leg < ROBOT_LEG_COUNT; leg += 2U)
        {
            if ((fabsf(targets[1].foot[leg].x - targets[0].foot[leg].x) > 1.0e-6f) ||
                (fabsf(targets[1].foot[leg].y - targets[0].foot[leg].y) > 1.0e-6f) ||
                (fabsf(targets[1].foot[leg].z - targets[0].foot[leg].z) > 1.0e-6f))
            {
                return false;  // 재개 첫 주기부터 위상 끝까지 기존 XYZ 경로 이탈을 검출한다.
            }
            if ((cycle >= first_hold_cycle) &&
                (!candidate_before_hold || (leg == 4U)) &&
                (targets[1].foot[leg].z > previous.foot[leg].z + 1.0e-6f))
            {
                return false;  // 이미 하강하던 공중발이 HOLD 뒤 새 Swing으로 상승하는 오류를 검출한다.
            }
        }
    }

    return true;
}

/* 같은 최대 조종 입력이 감속 후 여러 위상을 끊김 없이 진행하는지 검사한다. */
static bool GaitTest_CheckLimitedContinuousWalk(void)
{
    GaitManager_Handle_t manager;       // 연속 보행 위상 상태를 저장한다.
    WorkspaceLimiter_Handle_t limiter;  // 위상별 작업공간 검사 상태를 저장한다.
    RobotBodyTwist_t candidate = {0};   // 계속 유지할 최대 동시 입력을 저장한다.
    RobotBodyTwist_t applied;           // 검사 뒤 실제 보행 속도를 저장한다.
    RobotGaitPhase_t gait;              // 이번 주기 다리 상태를 저장한다.
    RobotEuler_t posture = {0};         // 평지 수평 자세를 저장한다.
    bool contact[ROBOT_LEG_COUNT];      // 지지와 착지를 보장할 접촉을 저장한다.
    bool accepted;                     // 원본 입력 채택 여부를 저장한다.
    bool first_validated = false;      // 최초 감속 검사 완료 여부를 저장한다.
    float saved_scale = 0.0f;          // 최초 성공한 축척을 저장한다.
    uint32_t preview_count = 0U;       // 교대로 검사한 Tripod 수를 저장한다.
    uint32_t retry_count = 0U;         // 최초 감속의 추가 대기 주기를 저장한다.
    uint32_t cycle;                    // 전체 보행 제어 주기를 저장한다.
    uint32_t leg;                      // 이상적 접촉을 설정할 다리 번호를 저장한다.
    const uint32_t cycle_limit =
        (uint32_t)(6.0f * ROBOT_GAIT_PHASE_TIME_S / ROBOT_CONTROL_PERIOD_S) +
        ROBOT_GAIT_START_DELAY_CYCLES + 32U;  // 여러 위상 완료를 기다릴 한계를 정한다.

    GaitManager_Init(&manager);                        // 최초 보행 대기를 준비한다.
    WorkspaceLimiter_Init(&limiter);                   // 명령과 검사 이력을 제거한다.
    candidate.vx = ROBOT_MAX_LINEAR_SPEED_MPS;          // 최대 전진 속도를 유지한다.
    candidate.vy = ROBOT_MAX_LATERAL_SPEED_MPS;         // 최대 횡이동 속도를 유지한다.
    candidate.wz = ROBOT_MAX_YAW_RATE_RADPS;            // 최대 회전 속도를 함께 유지한다.

    for (cycle = 0U; cycle < cycle_limit; ++cycle)
    {
        for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
        {
            const bool swing = ((leg % 2U) == (manager.phase_index % 2U));  // 현재 Tripod의 Swing 역할을 선택한다.

            contact[leg] = !manager.initialized || !swing ||
                           (manager.phase_time_s >= 0.95f * ROBOT_GAIT_PHASE_TIME_S);  // 지지발은 유지하고 Swing 끝에서 착지한다.
        }
        gait = GaitManager_StepContacts(&manager, true, true,
                                        limiter.phase_result_valid,
                                        limiter.phase_result_accepted,
                                        ROBOT_TRIPOD_NORMAL, 0.0f,
                                        contact, contact);  // 새 조종 입력 없이 기존 검사 결과로 위상을 진행한다.
        if (!gait.enabled_internal || gait.late_landing_hold ||
            gait.support_recovery_active ||
            ((preview_count > 0U) &&
             (manager.start_wait_count != ROBOT_GAIT_START_DELAY_CYCLES)))
        {
            return false;  // IK 감속 중 보행 해제나 100 ms 시작 대기 재진입을 검출한다.
        }

        if (gait.next_phase_preview)
        {
            const uint8_t expected_mask = ((preview_count % 2U) == 0U) ? 0x15U : 0x2AU;  // 다음 검사 그룹을 계산한다.

            if (gait.next_phase_swing_mask != expected_mask)
            {
                return false;  // 같은 Tripod만 다시 시작하는 오류를 검출한다.
            }
            preview_count++;  // 새로 요청한 위상 검사를 누적한다.
        }

        applied = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                        &gait, &posture, false,
                                        &accepted);  // 동일한 원본 입력을 매 주기 전달한다.
        if (accepted || (limiter.phase_result_valid && !limiter.phase_result_accepted))
        {
            return false;  // 위험한 원본 채택이나 해결 가능한 입력의 최종 거부를 검출한다.
        }
        if (limiter.preview_active)
        {
            if (first_validated || limiter.phase_result_valid || !gait.waiting_start)
            {
                return false;  // 최초 감속 이후 같은 명령에서 추가 검사 대기를 검출한다.
            }
            retry_count++;  // 최초 감속에 사용한 추가 제어 주기를 누적한다.
        }
        if (limiter.phase_result_valid && limiter.phase_result_accepted)
        {
            if (!first_validated)
            {
                saved_scale = applied.vx / candidate.vx;  // 최초 유효 보폭의 축척을 보존한다.
                first_validated = true;                  // 다음 위상부터 캐시 재사용을 검사한다.
            }
            if ((saved_scale <= 0.0f) || (saved_scale >= 1.0f) ||
                (fabsf(applied.vx - candidate.vx * saved_scale) > 1.0e-6f) ||
                (fabsf(applied.vy - candidate.vy * saved_scale) > 1.0e-6f) ||
                (fabsf(applied.wz - candidate.wz * saved_scale) > 1.0e-6f))
            {
                return false;  // 반복 위상에서 보폭 누적 축소나 방향 변화를 검출한다.
            }
        }
        if (manager.initialized && (manager.phase_index >= 4U))
        {
            return first_validated && (preview_count >= 5U) && (retry_count > 0U);
        }
    }

    return false;  // 같은 입력으로 여러 위상을 완료하지 못한 정지를 검출한다.
}

/* 두 번의 개별 보행 순회에서 한 발만 들고 나머지 접촉을 유지하는지 검사한다. */
static bool GaitTest_CheckWaveSequence(void)
{
    static const uint8_t order[ROBOT_LEG_COUNT] = {0U, 5U, 1U, 3U, 2U, 4U};  // 물리 다리 순서의 기대값을 정의한다.
    GaitManager_Handle_t manager;   // 시험할 보행 상태를 저장한다.
    RobotGaitPhase_t gait = {0};    // 직전 위상 출력을 저장한다.
    bool contact[ROBOT_LEG_COUNT];  // 다섯 지지발과 이동발의 접촉을 저장한다.
    uint32_t cycle;                 // 시험 제한 시간을 계산한다.
    uint32_t leg;                   // 확인할 다리 번호를 저장한다.
    uint32_t previews = 0U;         // 경로 검사를 거친 위상 수를 저장한다.

    GaitManager_Init(&manager);                         // 기본 정지 상태를 준비한다.
    GaitManager_SetPattern(&manager, ROBOT_GAIT_WAVE);  // 한 발 보행을 요청한다.
    for (cycle = 0U; cycle < 3000U; ++cycle)
    {
        memset(contact, 1, sizeof(contact));  // 지지발 접촉을 준비한다.
        if (manager.initialized && (manager.phase_time_s < 0.90f))
        {
            contact[order[manager.phase_index % ROBOT_LEG_COUNT]] = false;  // 이동 중인 발만 미접촉으로 둔다.
        }
        gait = GaitManager_StepContacts(&manager, true, true, true, true,
                                        ROBOT_TRIPOD_NORMAL, 0.0f,
                                        contact, contact);  // 접촉 기반의 실제 상태기를 진행한다.
        if (gait.next_phase_preview)
        {
            if ((gait.next_phase_pattern != ROBOT_GAIT_WAVE) ||
                (gait.next_phase_swing_mask != (uint8_t)(1U << order[previews % ROBOT_LEG_COUNT])))
            {
                return false;
            }
            previews++;  // 한 발 경로 검사 순서를 기록한다.
        }
        if (manager.initialized)
        {
            for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
            {
                if ((leg != order[manager.phase_index % ROBOT_LEG_COUNT]) &&
                    (gait.state[leg] != ROBOT_LEG_STANCE))
                {
                    return false;  // 이동발 외의 추가 이륙을 검출한다.
                }
            }
            if (gait.startup_phase != (manager.phase_index < ROBOT_LEG_COUNT))
            {
                return false;  // 첫 여섯 위상만 완만한 출발을 적용하는지 확인한다.
            }
            if (manager.phase_index >= 12U)
            {
                return previews == 13U;  // 두 순회 뒤 다음 발도 검사했는지 확인한다.
            }
        }
    }
    return false;
}

/* 단일 발 이동 중 정지·패턴 변경과 접촉 누락을 검사한다. */
static bool GaitTest_CheckWaveTransitions(void)
{
    GaitManager_Handle_t manager;   // 모드 전환 상태를 저장한다.
    RobotGaitPhase_t gait = {0};    // 이번 보행 출력을 저장한다.
    bool contact[ROBOT_LEG_COUNT];  // 명시적인 발 접촉을 저장한다.
    uint32_t cycle;                 // 대기와 진행 시간을 저장한다.
    uint32_t scenario;              // 정지와 양방향 전환을 구분한다.

    for (scenario = 0U; scenario < 3U; ++scenario)
    {
        const RobotGaitPattern_t initial = (scenario == 2U) ? ROBOT_GAIT_TRIPOD : ROBOT_GAIT_WAVE;  // 시작 패턴을 선택한다.
        const RobotGaitPattern_t requested = (initial == ROBOT_GAIT_WAVE) ? ROBOT_GAIT_TRIPOD : ROBOT_GAIT_WAVE;  // 반대 패턴을 선택한다.

        GaitManager_Init(&manager);                    // 각 시험을 독립적인 정지에서 시작한다.
        GaitManager_SetPattern(&manager, initial);     // 시험할 시작 패턴을 요청한다.
        memset(contact, 1, sizeof(contact));            // 출발 전 여섯 발의 접촉을 준비한다.
        for (cycle = 0U; cycle < 25U; ++cycle)
        {
            gait = GaitManager_StepContacts(&manager, true, true, true, true,
                                            ROBOT_TRIPOD_NORMAL, 0.0f, contact, contact);  // 첫 경로 검사 후 이륙한다.
        }
        if (!manager.initialized || (gait.gait_pattern != initial))
        {
            return false;
        }
        if (scenario != 0U)
        {
            GaitManager_SetPattern(&manager, requested);  // 공중에서 반대 패턴을 예약한다.
        }
        for (cycle = 0U; cycle < 220U; ++cycle)
        {
            contact[0] = false;  // 현재 첫 이동발의 착지를 지연한다.
            if (initial == ROBOT_GAIT_TRIPOD)
            {
                contact[2] = false;  // 첫 Tripod의 둘째 발도 이동시킨다.
                contact[4] = false;  // 첫 Tripod의 셋째 발도 이동시킨다.
            }
            gait = GaitManager_StepContacts(&manager, true, scenario != 0U, true, true,
                                            ROBOT_TRIPOD_NORMAL, 0.0f, contact, contact);  // 미접촉 상태에서 정지 또는 전환을 요청한다.
            if ((gait.gait_pattern != initial) || (manager.phase_index != 0U) || !gait.enabled_internal)
            {
                return false;  // 착지 전 정지 확정이나 다음 발 이륙을 검출한다.
            }
        }
        memset(contact, 1, sizeof(contact));  // 현재 이동발의 착지를 확정한다.
        gait = GaitManager_StepContacts(&manager, true, scenario != 0U, true, true,
                                        ROBOT_TRIPOD_NORMAL, 0.0f, contact, contact);  // 착지 경계에서 이전 패턴을 끝낸다.
        if (gait.enabled_internal)
        {
            return false;  // 전환 전 또는 정지 후 추가 이륙을 검출한다.
        }
        if (scenario == 0U)
        {
            continue;
        }
        for (cycle = 0U; cycle < 25U; ++cycle)
        {
            gait = GaitManager_StepContacts(&manager, true, true, false, false,
                                            ROBOT_TRIPOD_NORMAL, 0.0f, contact, contact);  // 새 패턴의 검사 결과를 일부러 지연한다.
            if (manager.initialized || (gait.gait_pattern != requested) || !gait.waiting_start)
            {
                return false;  // 이전 경로 검사 결과로 새 패턴이 출발하는지 확인한다.
            }
        }
        gait = GaitManager_StepContacts(&manager, true, true, true, true,
                                        ROBOT_TRIPOD_NORMAL, 0.0f, contact, contact);  // 새 패턴의 경로 검사만 승인한다.
        if (!manager.initialized || (gait.gait_pattern != requested))
        {
            return false;
        }
    }

    GaitManager_Init(&manager);                         // 접촉 누락 시험을 준비한다.
    GaitManager_SetPattern(&manager, ROBOT_GAIT_WAVE);  // 한 발 보행을 요청한다.
    memset(contact, 1, sizeof(contact));                 // 출발 자세를 준비한다.
    contact[3] = false;                                  // 지지할 발 하나의 접촉을 제거한다.
    for (cycle = 0U; cycle < 50U; ++cycle)
    {
        gait = GaitManager_StepContacts(&manager, true, true, true, true,
                                        ROBOT_TRIPOD_NORMAL, 0.0f, contact, contact);  // 지지발이 부족한 시작을 시도한다.
        if (manager.initialized || gait.next_phase_preview)
        {
            return false;
        }
    }
    return true;
}

/* 개별 보행의 지지발 연속 이동과 착지 지연 중 위치 유지를 검사한다. */
static bool GaitTest_CheckWaveTrajectory(void)
{
    FootTrajectory_Handle_t trajectory;                  // 여섯 발의 연속 목표를 저장한다.
    RobotDroneOutput_t drone = {0};                      // 수동 보행 조건을 준비한다.
    RobotGaitPhase_t gait = {0};                         // 단일 이동발의 진행률을 저장한다.
    RobotBodyTwist_t twist = {0.02f, 0.0f, 0.0f, 0.05f};  // 전진과 회전을 함께 요청한다.
    RobotEuler_t posture = {0};                          // 수평 자세를 준비한다.
    RobotVec3_t base[ROBOT_LEG_COUNT];                    // 시험의 시작 지지점을 저장한다.
    RobotFootTargets_t feet;                             // 계산된 목표를 저장한다.
    uint32_t cycle;                                      // 한 위상과 착지 대기를 진행한다.
    uint32_t leg;                                        // 확인할 지지발을 선택한다.

    FootTrajectory_Init(&trajectory);         // 중립 발 위치를 준비한다.
    LegKinematics_GetBaseFeet(base);           // 실제 기본 발 위치를 읽는다.
    drone.manual_enable = true;              // 수동 보행을 선택한다.
    drone.body_control_enable = true;        // 몸체 이동 궤적을 활성화한다.
    drone.tripod_enable = true;              // 정상 보행을 활성화한다.
    gait.gait_pattern = ROBOT_GAIT_WAVE;      // 한 발 보행 궤적을 선택한다.
    gait.enabled_internal = true;            // 내부 보행을 허가한다.
    gait.state[0] = ROBOT_LEG_SWING;          // 첫 발만 공중 이동시킨다.
    for (cycle = 0U; cycle <= 220U; ++cycle)
    {
        const float time = fminf((float)cycle * ROBOT_CONTROL_PERIOD_S, ROBOT_GAIT_PHASE_TIME_S);  // 착지 지연 이후 진행률을 고정한다.
        for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
        {
            gait.progress[leg] = time / ROBOT_GAIT_PHASE_TIME_S;  // 모든 발의 동일한 위상 시각을 준비한다.
        }
        feet = FootTrajectory_Step(&trajectory, &twist, &drone, &gait, &posture);  // 실제 보행 궤적을 생성한다.
        for (leg = 1U; leg < ROBOT_LEG_COUNT; ++leg)
        {
            const RobotVec3_t expected = StanceTrajectory_Advance(&base[leg], &twist, time);  // 누적 시간의 지지점 기대값을 계산한다.
            if ((fabsf(feet.foot[leg].x - expected.x) > 0.00001f) ||
                (fabsf(feet.foot[leg].y - expected.y) > 0.00001f) ||
                (fabsf(feet.foot[leg].z - expected.z) > 0.00001f))
            {
                return false;  // 위상 시작 점프나 Late Landing 중 추가 적분을 검출한다.
            }
        }
    }
    return true;
}

/* 접촉·경로 검사·실제 궤적을 연결해 전후진과 회전의 한 발 보행을 검사한다. */
static bool GaitTest_CheckWaveIntegrated(void)
{
    static const uint8_t order[ROBOT_LEG_COUNT] = {0U, 5U, 1U, 3U, 2U, 4U};  // 다리별 착지 센서 모형의 순서를 정의한다.
    GaitManager_Handle_t manager;         // 실제 보행 상태기를 저장한다.
    FootTrajectory_Handle_t trajectory;  // 실제 발 목표를 저장한다.
    WorkspaceLimiter_Handle_t limiter;  // 다음 발의 경로 검사 상태를 저장한다.
    RobotDroneOutput_t drone = {0};      // 수동 보행을 준비한다.
    RobotEuler_t posture = {0};          // 수평 자세를 준비한다.
    RobotVec3_t base[ROBOT_LEG_COUNT];    // 평지 접촉 높이를 저장한다.
    bool contact[ROBOT_LEG_COUNT];       // 궤적 높이로 만든 접촉을 저장한다.
    uint32_t scenario;                   // 전진·후진·회전을 구분한다.
    uint32_t cycle;                      // 시뮬레이션 제어 주기를 저장한다.
    uint32_t leg;                        // 발별 유효성을 확인한다.

    drone.manual_enable = true;          // 수동 보행을 활성화한다.
    drone.body_control_enable = true;    // 보행 목표 생성을 활성화한다.
    drone.tripod_enable = true;          // 짐벌 보행 명령을 유지한다.
    drone.gait_pattern = ROBOT_GAIT_WAVE; // 단일 발 패턴을 요청한다.
    LegKinematics_GetBaseFeet(base);      // 실제 기본 자세를 읽는다.
    for (scenario = 0U; scenario < 3U; ++scenario)
    {
        RobotBodyTwist_t candidate = {0};  // 감속된 보행 속도를 준비한다.
        bool completed = false;           // 정상 순회 완료를 기록한다.

        candidate.vx = (scenario == 0U) ? 0.02f : ((scenario == 1U) ? -0.02f : 0.0f);  // 전후 최대 감속 입력을 선택한다.
        candidate.wz = (scenario == 2U) ? ROBOT_MAX_YAW_RATE_RADPS * ROBOT_WAVE_SPEED_SCALE : 0.0f;  // 회전 최대 감속 입력을 선택한다.
        GaitManager_Init(&manager);                         // 첫 위상을 준비한다.
        GaitManager_SetPattern(&manager, ROBOT_GAIT_WAVE);  // 한 발 보행을 요청한다.
        FootTrajectory_Init(&trajectory);                  // 기본 발 위치를 준비한다.
        WorkspaceLimiter_Init(&limiter);                    // 검사 결과를 초기화한다.
        for (cycle = 0U; cycle < 2400U; ++cycle)
        {
            RobotGaitPhase_t gait;    // 이번 위상 결과를 저장한다.
            RobotBodyTwist_t twist;  // 실제 적용 속도를 저장한다.
            RobotFootTargets_t feet; // 실제 발 목표를 저장한다.
            bool accepted;           // 명령 채택 여부를 저장한다.

            memset(contact, 1, sizeof(contact));  // 지지발 접촉을 유지한다.
            if (manager.initialized && !manager.landed[order[manager.phase_index % ROBOT_LEG_COUNT]])
            {
                const uint32_t swing = order[manager.phase_index % ROBOT_LEG_COUNT];  // 현재 이동발을 선택한다.
                contact[swing] = (manager.phase_time_s >= 0.8f) &&
                    (trajectory.memory[swing].z <= base[swing].z + 0.003f);  // 하강하는 발의 평지 접촉을 모형화한다.
            }
            gait = GaitManager_StepContacts(&manager, true, true,
                                            limiter.phase_result_valid, limiter.phase_result_accepted,
                                            ROBOT_TRIPOD_NORMAL, 0.0f, contact, contact);  // 실제 검사 결과로 다음 발 이륙을 결정한다.
            WorkspaceLimiter_SetFeet(&limiter, trajectory.memory, &trajectory.body_offset_m);  // 실제 시작점으로 검사한다.
            twist = WorkspaceLimiter_Gait(&limiter, &candidate, 0.0f, true,
                                           &gait, &posture, false, &accepted);  // 발 작업공간에 맞는 보폭을 적용한다.
            feet = FootTrajectory_Step(&trajectory, &twist, &drone, &gait, &posture);  // 여섯 발의 목표를 계산한다.
            if (gait.late_landing_hold || (limiter.phase_result_valid && !limiter.phase_result_accepted))
            {
                return false;  // 평지 명령에서 접촉 실패나 불가능한 경로를 검출한다.
            }
            if ((cycle % 10U) == 0U)
            {
                for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
                {
                    if (!LegKinematics_IsReachable((uint8_t)leg, &feet.foot[leg]))
                    {
                        return false;  // 실제 생성한 궤적의 관절 범위 위반을 검출한다.
                    }
                }
            }
            if (manager.initialized && (manager.phase_index >= 7U))
            {
                completed = true;  // 첫 순회와 정상 지지 속도의 다음 발까지 확인한다.
                break;
            }
        }
        if (!completed)
        {
            return false;
        }
    }
    return true;
}

/* 명시적 접촉 입력으로 Tripod 상태 전환과 짧은 Enable 변화를 검사한다. */
bool GaitTest_Run(void)
{
    GaitManager_Handle_t manager;                  // 시험용 Tripod 상태를 저장한다.
    RobotGaitPhase_t phase;                         // 한 주기 다리 상태를 저장한다.
    bool contact[ROBOT_LEG_COUNT];                  // 명시적 접촉 입력을 저장한다.
    const uint32_t early_cycles =
        (uint32_t)((ROBOT_GAIT_PHASE_TIME_S * ROBOT_EARLY_LANDING_PROGRESS) /
                   ROBOT_CONTROL_PERIOD_S) + 1U;  // 조기 착지 허용 직후 주기를 계산한다.
    uint32_t cycle;                                 // 진행할 제어 주기 번호를 저장한다.

    if (!GaitTest_CheckWaveIntegrated() ||
        !GaitTest_CheckWaveSequence() ||
        !GaitTest_CheckWaveTransitions() ||
        !GaitTest_CheckWaveTrajectory() ||
        !GaitTest_CheckImmediateContactHold() ||
        !GaitTest_CheckManualStanceHold() ||
        !GaitTest_CheckCommonZRecovery() ||
        !GaitTest_CheckLateLandingLimit() ||
        !GaitTest_CheckStartupPreview() ||
        !GaitTest_CheckPhaseTwistLatch() ||
        !GaitTest_CheckPreviewTransition() ||
        !GaitTest_CheckTwoStepCommandLatch() ||
        !GaitTest_CheckTouchdownConfirmation() ||
        !GaitTest_CheckSupportRecovery() ||
        !GaitTest_CheckSupportRecoveryTrajectory(false, false) ||
        !GaitTest_CheckSupportRecoveryTrajectory(true, false) ||
        !GaitTest_CheckSupportRecoveryTrajectory(false, true) ||
        !GaitTest_CheckLimitedContinuousWalk())
    {
        return false;
    }

    memset(contact, 0, sizeof(contact));  // Swing 비접촉 상태를 만든다.
    GaitManager_Init(&manager);           // Tripod 위상을 정지로 초기화한다.
    phase = GaitTest_StartNormal(&manager, contact);  // 정상 보행을 시작한다.
    if (!phase.enabled_internal)
    {
        return false;
    }

    for (cycle = 0U; cycle < early_cycles; ++cycle)
    {
        phase = GaitManager_Step(&manager, true, true, false, false,
                                 ROBOT_TRIPOD_NORMAL,
                                 0.0f, contact);  // Swing 50% 이후까지 진행한다.
    }
    contact[0] = true;  // 1번 Swing 다리의 Early Landing 입력을 넣는다.
    phase = GaitManager_Step(&manager, true, true, false, false,
                             ROBOT_TRIPOD_NORMAL,
                             0.0f, contact);  // 접촉 적응을 실행한다.
    if (phase.state[0] == ROBOT_LEG_SWING)
    {
        return false;
    }

    phase = GaitManager_Step(&manager, true, false, false, false,
                             ROBOT_TRIPOD_NORMAL,
                             0.0f, contact);  // 한 주기 Enable OFF를 넣는다.
    phase = GaitManager_Step(&manager, true, true, false, false,
                             ROBOT_TRIPOD_NORMAL,
                             0.0f, contact);  // 즉시 Enable을 복구한다.
    return phase.enabled_internal;  // 짧은 OFF에서 위상이 유지되는지 확인한다.
}
