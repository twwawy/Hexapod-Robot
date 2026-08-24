#ifndef BODY_POSITION_ESTIMATOR_H
#define BODY_POSITION_ESTIMATOR_H

#include "common/robot_types.h"

typedef struct
{
    RobotVec3_t body_position_world;              // 최근 몸체 위치 추정값을 저장한다.
    RobotVec3_t anchor_world[ROBOT_LEG_COUNT];    // Stance 진입 발 Anchor를 저장한다.
    bool was_valid[ROBOT_LEG_COUNT];              // 이전 Stance 유효성을 저장한다.
    uint8_t slip_suspect[ROBOT_LEG_COUNT];        // Slip 의심 연속 횟수를 저장한다.
    bool slip_latched[ROBOT_LEG_COUNT];           // 다리별 Slip Latch를 저장한다.
} BodyPositionEstimator_Handle_t;

typedef struct
{
    RobotVec3_t position_world;  // 추정한 몸체 위치를 저장한다.
    uint8_t valid_leg_count;     // 평균에 사용한 다리 수를 저장한다.
    uint8_t slip_leg_mask;       // Slip 다리를 비트로 저장한다.
} BodyPositionEstimator_Output_t;

void BodyPositionEstimator_Init(BodyPositionEstimator_Handle_t *handle);  // Anchor와 Slip 상태를 초기화한다.

BodyPositionEstimator_Output_t BodyPositionEstimator_Step(
    BodyPositionEstimator_Handle_t *handle,
    const float joint_angle_rad[ROBOT_JOINT_COUNT],
    const RobotGaitPhase_t *gait,
    const bool contact[ROBOT_LEG_COUNT],
    const RobotEuler_t *attitude_rad);  // Stance FK로 몸체 위치와 Slip을 추정한다.

#endif
