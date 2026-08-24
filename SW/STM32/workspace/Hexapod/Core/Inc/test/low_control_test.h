#ifndef LOW_CONTROL_TEST_H
#define LOW_CONTROL_TEST_H

#include "low_control/relay.h"
#include "low_control/servo_pwm.h"

#include <stdbool.h>

typedef struct
{
    ServoPwm_Handle_t *servo;  // 시험할 PWM 출력을 참조한다.
    bool initialized;          // 안전 초기화 여부를 저장한다.
} LowControlTest_t;

bool LowControlTest_Init(LowControlTest_t *test,
                         ServoPwm_Handle_t *servo);  // PWM 중립과 릴레이 OFF를 준비한다.
bool LowControlTest_WriteNeutral(LowControlTest_t *test);  // 18개 PWM을 관절 0도 목표로 갱신한다.
bool LowControlTest_SetRelay(LowControlTest_t *test,
                             Relay_Channel_t relay,
                             bool on);  // 사용자가 확인한 한 릴레이만 명시적으로 설정한다.
void LowControlTest_Stop(LowControlTest_t *test);  // 릴레이를 끄고 PWM을 정지한다.

#endif
