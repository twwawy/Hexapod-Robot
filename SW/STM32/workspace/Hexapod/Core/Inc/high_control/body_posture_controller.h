#ifndef BODY_POSTURE_CONTROLLER_H
#define BODY_POSTURE_CONTROLLER_H

#include "common/robot_types.h"

typedef struct
{
    RobotEuler_t command_rad;       // 마지막 정상 자세 명령을 저장한다.
    RobotEuler_t integral;          // 자세 PI 적분값을 저장한다.
    float correction_yaw_base_rad;  // 보정 모드 진입 Heading을 저장한다.
    bool previous_manual;           // 이전 수동 모드 상태를 저장한다.
    bool previous_correction;       // 이전 보정 모드 상태를 저장한다.
} BodyPostureController_Handle_t;

typedef struct
{
    RobotFootTargets_t targets;  // 자세 역회전을 적용한 발 위치를 저장한다.
    RobotEuler_t command_rad;    // 실제 채택한 자세 명령을 저장한다.
    bool accepted;               // 자세 후보 채택 여부를 저장한다.
} BodyPostureController_Output_t;

void BodyPostureController_Init(BodyPostureController_Handle_t *handle);  // 자세 PI 상태를 초기화한다.

BodyPostureController_Output_t BodyPostureController_Step(
    BodyPostureController_Handle_t *handle,
    const RobotVec3_t feet_body[ROBOT_LEG_COUNT],
    const RobotDroneOutput_t *drone,
    const RobotEuler_t *measured_rad,
    bool reset_command);  // 자세 PI와 작업공간 채택을 계산한다.

#endif
