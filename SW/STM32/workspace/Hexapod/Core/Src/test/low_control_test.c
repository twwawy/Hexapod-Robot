#include "test/low_control_test.h"

#include <stddef.h>
#include <string.h>

/* 릴레이 OFF와 PWM 중립으로 전기적 출력 시험을 준비한다. */
bool LowControlTest_Init(LowControlTest_t *test,
                         ServoPwm_Handle_t *servo)
{
    if ((test == NULL) || (servo == NULL))
    {
        return false;
    }

    memset(test, 0, sizeof(*test));  // 이전 시험 상태를 제거한다.
    test->servo = servo;             // 시험할 PWM 상태를 연결한다.
    Relay_Init();                    // 서보 전원을 모두 차단한다.
    test->initialized = true;        // 안전 초기화를 표시한다.
    return true;
}

/* 18개 관절에 0도 명령을 보내 PWM 배선만 확인한다. */
bool LowControlTest_WriteNeutral(LowControlTest_t *test)
{
    float angle_rad[ROBOT_JOINT_COUNT] = {0.0f};  // 모든 관절의 0도 목표를 만든다.

    return (test != NULL) && test->initialized &&
           ServoPwm_WriteAngles(test->servo, angle_rad);  // 보정표 기준 중립 PWM을 출력한다.
}

/* 명시적으로 선택한 릴레이 하나만 켜거나 끈다. */
bool LowControlTest_SetRelay(LowControlTest_t *test,
                             Relay_Channel_t relay,
                             bool on)
{
    if ((test == NULL) || !test->initialized ||
        ((uint32_t)relay >= (uint32_t)RELAY_CHANNEL_COUNT))
    {
        return false;
    }

    Relay_Set(relay, on);  // 사용자가 선택한 한 채널만 변경한다.
    return Relay_IsOn(relay) == on;
}

/* 전기적 출력 시험을 안전 상태로 종료한다. */
void LowControlTest_Stop(LowControlTest_t *test)
{
    Relay_AllOff();  // 모든 서보 전원을 차단한다.
    if ((test != NULL) && (test->servo != NULL))
    {
        ServoPwm_Stop(test->servo);  // 모든 PWM 출력을 정지한다.
        test->initialized = false;   // 시험 종료를 표시한다.
    }
}
