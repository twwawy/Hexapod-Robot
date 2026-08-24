#include "test/rc_command_generator.h"

#include "common/robot_config.h"

#include <stddef.h>
#include <string.h>

#define RC_COMMAND_RAMP_PER_SECOND 5000.0f
#define RC_COMMAND_RAMP_STEP       ((int16_t)(RC_COMMAND_RAMP_PER_SECOND * ROBOT_CONTROL_PERIOD_S))

/* 임시 짐벌 목표를 프로젝트 범위로 제한한다. */
static int16_t RcCommandGenerator_Clamp(int16_t value)
{
    if (value < -1000)
    {
        return -1000;
    }
    if (value > 1000)
    {
        return 1000;
    }
    return value;
}

/* 한 축을 목표값까지 정해진 최대 변화량으로 이동한다. */
static int16_t RcCommandGenerator_Move(int16_t current, int16_t target)
{
    const int32_t difference = (int32_t)target - (int32_t)current;  // 남은 입력 차이를 계산한다.

    if (difference > RC_COMMAND_RAMP_STEP)
    {
        return (int16_t)(current + RC_COMMAND_RAMP_STEP);  // 양의 방향 변화량을 제한한다.
    }
    if (difference < -RC_COMMAND_RAMP_STEP)
    {
        return (int16_t)(current - RC_COMMAND_RAMP_STEP);  // 음의 방향 변화량을 제한한다.
    }
    return target;
}

/* 임시 조종기 출력을 연결된 중립 상태로 초기화한다. */
void RcCommandGenerator_Init(RcCommandGenerator_t *generator)
{
    if (generator == NULL)
    {
        return;
    }

    memset(generator, 0, sizeof(*generator));  // 이전 Ramp 상태를 제거한다.
    generator->output.connected = true;        // 임시 시험 입력 연결을 표시한다.
    generator->output.motion_armed = true;     // 시험 동작을 허가한다.
    generator->target = generator->output;     // 초기 목표도 중립으로 둔다.
}

/* -1000~1000 범위의 다음 시험 목표와 스위치 상태를 저장한다. */
void RcCommandGenerator_SetTarget(RcCommandGenerator_t *generator,
                                  const RobotUserCommand_t *target)
{
    if ((generator == NULL) || (target == NULL))
    {
        return;
    }

    generator->target = *target;  // 시험 시퀀스가 정한 스위치 목표를 저장한다.
    generator->target.throttle = RcCommandGenerator_Clamp(target->throttle);  // Throttle 목표를 제한한다.
    generator->target.yaw = RcCommandGenerator_Clamp(target->yaw);            // Yaw 목표를 제한한다.
    generator->target.roll = RcCommandGenerator_Clamp(target->roll);          // Roll 목표를 제한한다.
    generator->target.pitch = RcCommandGenerator_Clamp(target->pitch);        // Pitch 목표를 제한한다.
}

/* 네 짐벌을 5000 raw/s로 Ramp하고 스위치는 즉시 적용한다. */
RobotUserCommand_t RcCommandGenerator_Step(RcCommandGenerator_t *generator)
{
    RobotUserCommand_t empty;  // 잘못된 입력의 안전 출력을 저장한다.

    memset(&empty, 0, sizeof(empty));  // 기본 안전 출력을 0으로 만든다.
    if (generator == NULL)
    {
        return empty;
    }

    generator->output.throttle = RcCommandGenerator_Move(generator->output.throttle,
                                                          generator->target.throttle);  // Throttle을 Ramp한다.
    generator->output.yaw = RcCommandGenerator_Move(generator->output.yaw,
                                                     generator->target.yaw);            // Yaw를 Ramp한다.
    generator->output.roll = RcCommandGenerator_Move(generator->output.roll,
                                                      generator->target.roll);          // Roll을 Ramp한다.
    generator->output.pitch = RcCommandGenerator_Move(generator->output.pitch,
                                                       generator->target.pitch);        // Pitch를 Ramp한다.
    generator->output.sa = generator->target.sa;  // SA 시험 상태를 적용한다.
    generator->output.sb = generator->target.sb;  // SB 시험 상태를 적용한다.
    generator->output.sc = generator->target.sc;  // SC 시험 상태를 적용한다.
    generator->output.sd = generator->target.sd;  // SD 시험 상태를 적용한다.
    generator->output.se = generator->target.se;  // SE 시험 상태를 적용한다.
    generator->output.connected = true;           // 임시 연결을 유지한다.
    generator->output.motion_armed = true;        // 시험 동작 허가를 유지한다.
    return generator->output;
}
