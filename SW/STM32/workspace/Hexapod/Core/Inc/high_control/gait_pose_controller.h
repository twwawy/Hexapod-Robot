#ifndef GAIT_POSE_CONTROLLER_H
#define GAIT_POSE_CONTROLLER_H

#include "common/robot_types.h"

typedef struct
{
    float x_reference_m;                  // World X 기준값을 저장한다.
    float y_reference_m;                  // World Y 기준값을 저장한다.
    float x_integral;                     // X 위치 적분값을 저장한다.
    float y_integral;                     // Y 위치 적분값을 저장한다.
    float yaw_integral;                   // Yaw 적분값을 저장한다.
    float previous_yaw_feedback_radps;  // 직전 Heading 보정을 저장한다.
    RobotBodyTwist_t previous;            // Rate Limit 이전 명령을 저장한다.
    bool previous_manual;                 // 이전 수동 모드 상태를 저장한다.
} GaitPoseController_Handle_t;

typedef struct
{
    RobotBodyTwist_t twist;      // 최종 Body Twist 후보를 저장한다.
    float x_reference_m;         // World X 기준값을 저장한다.
    float y_reference_m;         // World Y 기준값을 저장한다.
    RobotVec3_t feedback_world;  // World 위치 Feedback을 저장한다.
    float yaw_feedback_radps;    // Yaw Feedback을 저장한다.
} GaitPoseController_Output_t;

void GaitPoseController_Init(GaitPoseController_Handle_t *handle);  // PI와 Rate Limit 상태를 초기화한다.

GaitPoseController_Output_t GaitPoseController_Step(
    GaitPoseController_Handle_t *handle,
    bool reset_command,
    const RobotDroneOutput_t *drone,
    const RobotVec3_t *body_position_world,
    uint8_t valid_leg_count,
    float yaw_measured_rad);  // 위치·Heading PI와 사용자 명령을 결합한다.

#endif
