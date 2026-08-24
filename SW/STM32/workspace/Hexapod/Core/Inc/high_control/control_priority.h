#ifndef CONTROL_PRIORITY_H
#define CONTROL_PRIORITY_H

#include "common/robot_types.h"

typedef enum
{
    CONTROL_SUPERVISOR_LANDED = 0,  // 완전 착지 상태를 나타낸다.
    CONTROL_SUPERVISOR_STANDING,    // 서기 진행 상태를 나타낸다.
    CONTROL_SUPERVISOR_READY,       // 동작 준비 상태를 나타낸다.
    CONTROL_SUPERVISOR_LANDING,     // 착지 진행 상태를 나타낸다.
    CONTROL_SUPERVISOR_FAULT,       // Fault 상태를 나타낸다.
    CONTROL_SUPERVISOR_KILL         // Kill 상태를 나타낸다.
} ControlPriority_Supervisor_t;

typedef struct
{
    ControlPriority_Supervisor_t supervisor;  // 상위 상태를 저장한다.
    bool stand_command_armed;                  // 서기 재입력 허가를 저장한다.
    bool motion_armed;                         // READY 입력 허가를 저장한다.
    float neutral_time_s;                      // READY 중립 유지 시간을 저장한다.
} ControlPriority_Handle_t;

void ControlPriority_Init(ControlPriority_Handle_t *handle);  // 상위 상태를 LANDED로 초기화한다.

RobotPriorityOutput_t ControlPriority_Step(ControlPriority_Handle_t *handle,
                                           const RobotUserCommand_t *user,
                                           bool stand_done,
                                           bool landing_done,
                                           const RobotSafetyOutput_t *safety);  // 우선순위와 전달 명령을 계산한다.

#endif
