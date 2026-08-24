#ifndef RC_COMMAND_GENERATOR_H
#define RC_COMMAND_GENERATOR_H

#include "common/robot_types.h"

#include <stdint.h>

typedef struct
{
    RobotUserCommand_t output;  // 현재 Ramp 출력을 저장한다.
    RobotUserCommand_t target;  // 다음 최대값 목표를 저장한다.
} RcCommandGenerator_t;

void RcCommandGenerator_Init(RcCommandGenerator_t *generator);  // 임시 조종기 명령을 중립으로 초기화한다.
void RcCommandGenerator_SetTarget(RcCommandGenerator_t *generator,
                                  const RobotUserCommand_t *target);  // 다음 최대값 시험 목표를 설정한다.
RobotUserCommand_t RcCommandGenerator_Step(RcCommandGenerator_t *generator);  // 0.4초 왕복 속도로 한 주기 Ramp한다.

#endif
