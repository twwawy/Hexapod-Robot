#ifndef FOOT_TRAJECTORY_H
#define FOOT_TRAJECTORY_H

#include "common/robot_types.h"

typedef struct
{
    RobotVec3_t body_offset_m;                       // 보정 모드 몸체 Offset을 저장한다.
    RobotVec3_t memory[ROBOT_LEG_COUNT];             // 다리별 연속 발 위치를 저장한다.
    RobotVec3_t swing_start[ROBOT_LEG_COUNT];        // 사용자 정의 Swing 시작점을 저장한다.
    RobotVec3_t recovery_start[ROBOT_LEG_COUNT];     // 복구 Swing 시작점을 저장한다.
    RobotLegState_t previous_state[ROBOT_LEG_COUNT]; // 이전 다리 상태를 저장한다.
    bool adapted_stance[ROBOT_LEG_COUNT];            // 접촉 적응 Stance 여부를 저장한다.
    bool custom_swing[ROBOT_LEG_COUNT];              // 연속 시작점 사용 여부를 저장한다.
    bool initialized;                                // 기본 발 위치 초기화 여부를 저장한다.
} FootTrajectory_Handle_t;

void FootTrajectory_Init(FootTrajectory_Handle_t *handle);  // 발 위치와 보정 Offset을 초기화한다.

RobotFootTargets_t FootTrajectory_Step(FootTrajectory_Handle_t *handle,
                                       const RobotBodyTwist_t *twist,
                                       const RobotDroneOutput_t *drone,
                                       const RobotGaitPhase_t *gait,
                                       const RobotEuler_t *posture_rad);  // 여섯 발의 연속 궤적을 계산한다.

#endif
