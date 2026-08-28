#ifndef FOOT_TRAJECTORY_H
#define FOOT_TRAJECTORY_H

#include "common/robot_types.h"

typedef struct
{
    RobotVec3_t body_offset_m;                       // 보정 모드 몸체 Offset을 저장한다.
    RobotVec3_t memory[ROBOT_LEG_COUNT];             // 다리별 연속 발 위치를 저장한다.
    RobotVec3_t swing_start[ROBOT_LEG_COUNT];        // 사용자 정의 Swing 시작점을 저장한다.
    RobotVec3_t recovery_start[ROBOT_LEG_COUNT];     // 복구 Swing 시작점을 저장한다.

    RobotBodyTwist_t phase_twist;                     // 현재 위상에 고정한 보행 속도를 저장한다.

    float landing_target_z[ROBOT_LEG_COUNT];         // 다리별 정상 착지 Z를 저장한다.
    float landing_z_error[ROBOT_LEG_COUNT];          // 다리별 PWM 명령 착지 Z 오차를 저장한다.
    float common_z_recovery_remaining[2];            // 두 Tripod의 남은 Z 복구량을 저장한다.
    float common_z_recovery_total[2];                // 두 Tripod의 S-curve 전체 복구량을 저장한다.
    float common_z_recovery_progress[2];             // 두 Tripod의 S-curve 진행률을 저장한다.
    uint32_t touchdown_time_ms[ROBOT_LEG_COUNT];     // 다리별 접촉 검출 시각을 저장한다.
    RobotLegState_t previous_state[ROBOT_LEG_COUNT]; // 이전 다리 상태를 저장한다.
    uint8_t previous_swing_mask;                     // 직전 Swing 다리 비트를 저장한다.

    bool adapted_stance[ROBOT_LEG_COUNT];            // 접촉 적응 Stance 여부를 저장한다.
    bool custom_swing[ROBOT_LEG_COUNT];              // 연속 시작점 사용 여부를 저장한다.
    bool landing_z_error_valid[ROBOT_LEG_COUNT];     // 다리별 착지 Z 오차 수집 여부를 저장한다.
    bool touchdown_pending[ROBOT_LEG_COUNT];         // 안정 후 PWM 명령 FK 대기를 저장한다.
    bool phase_twist_valid;                          // 현재 위상 속도 고정 여부를 저장한다.
    bool initialized;                                // 기본 발 위치 초기화 여부를 저장한다.
} FootTrajectory_Handle_t;

void FootTrajectory_Init(FootTrajectory_Handle_t *handle);  // 발 위치와 보정 Offset을 초기화한다.

bool FootTrajectory_LatchTouchdown(FootTrajectory_Handle_t *handle,
                                   uint8_t leg,
                                   const RobotVec3_t *commanded_foot_body,
                                   const RobotEuler_t *posture_rad,
                                   uint32_t now_ms);  // 접촉 순간 PWM 명령 발 위치를 고정한다.

void FootTrajectory_UpdateCommandedLanding(FootTrajectory_Handle_t *handle,
                                            const float joint_angle_rad[ROBOT_JOINT_COUNT],
                                            const RobotGaitPhase_t *gait,
                                            const RobotEuler_t *posture_rad,
                                            bool common_z_enable,
                                            uint32_t now_ms);  // 안정 후 PWM 명령 FK로 착지 Z 오차를 계산한다.

RobotFootTargets_t FootTrajectory_Step(FootTrajectory_Handle_t *handle,
                                       const RobotBodyTwist_t *twist,
                                       const RobotDroneOutput_t *drone,
                                       const RobotGaitPhase_t *gait,
                                       const RobotEuler_t *posture_rad);  // 여섯 발의 연속 궤적을 계산한다.

#endif
