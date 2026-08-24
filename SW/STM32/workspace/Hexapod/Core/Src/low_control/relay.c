#include "low_control/relay.h"

#include "main.h"

#include <stddef.h>

typedef struct
{
    GPIO_TypeDef *port;  // 릴레이 GPIO 포트를 저장한다.
    uint16_t pin;        // 릴레이 GPIO 핀을 저장한다.
} Relay_Pin_t;

static const Relay_Pin_t relay_pins[RELAY_CHANNEL_COUNT] =
{
    {INA1_GPIO_Port, INA1_Pin},  // 오른쪽 A 릴레이를 연결한다.
    {INB1_GPIO_Port, INB1_Pin},  // 오른쪽 B 릴레이를 연결한다.
    {INC1_GPIO_Port, INC1_Pin},  // 오른쪽 C 릴레이를 연결한다.
    {INA2_GPIO_Port, INA2_Pin},  // 왼쪽 A 릴레이를 연결한다.
    {INB2_GPIO_Port, INB2_Pin},  // 왼쪽 B 릴레이를 연결한다.
    {INC2_GPIO_Port, INC2_Pin}   // 왼쪽 C 릴레이를 연결한다.
};

static uint8_t relay_state_mask;  // 최근 릴레이 출력을 비트로 저장한다.

/* 릴레이 채널 번호의 유효성을 확인한다. */
static bool Relay_IsValidChannel(Relay_Channel_t channel)
{
    return (uint32_t)channel < (uint32_t)RELAY_CHANNEL_COUNT;  // 배열 접근 가능 여부를 반환한다.
}

/* 모든 릴레이를 안전한 OFF로 초기화한다. */
void Relay_Init(void)
{
    relay_state_mask = 0U;  // 내부 상태를 먼저 초기화한다.
    Relay_AllOff();         // 실제 출력을 모두 차단한다.
}

/* 한 릴레이의 Active High 출력을 설정한다. */
void Relay_Set(Relay_Channel_t channel, bool on)
{
    if (!Relay_IsValidChannel(channel))
    {
        return;
    }

    HAL_GPIO_WritePin(relay_pins[channel].port,
                      relay_pins[channel].pin,
                      on ? GPIO_PIN_SET : GPIO_PIN_RESET);  // 선택한 GPIO를 출력한다.

    if (on)
    {
        relay_state_mask |= (uint8_t)(1U << (uint32_t)channel);   // ON 상태를 기록한다.
    }
    else
    {
        relay_state_mask &= (uint8_t)~(1U << (uint32_t)channel);  // OFF 상태를 기록한다.
    }
}

/* 한 릴레이를 켠다. */
void Relay_On(Relay_Channel_t channel)
{
    Relay_Set(channel, true);  // 선택한 서보 전원을 공급한다.
}

/* 한 릴레이를 끈다. */
void Relay_Off(Relay_Channel_t channel)
{
    Relay_Set(channel, false);  // 선택한 서보 전원을 차단한다.
}

/* 모든 릴레이를 켠다. */
void Relay_AllOn(void)
{
    uint32_t channel;  // 변경할 릴레이 번호를 저장한다.

    for (channel = 0U; channel < RELAY_CHANNEL_COUNT; ++channel)
    {
        Relay_On((Relay_Channel_t)channel);  // 현재 릴레이를 켠다.
    }
}

/* 모든 릴레이를 끈다. */
void Relay_AllOff(void)
{
    uint32_t channel;  // 변경할 릴레이 번호를 저장한다.

    for (channel = 0U; channel < RELAY_CHANNEL_COUNT; ++channel)
    {
        Relay_Off((Relay_Channel_t)channel);  // 현재 릴레이를 끈다.
    }
}

/* Kill이 없을 때만 요청한 릴레이 전원을 허용한다. */
void Relay_ApplySafety(bool relay_enable, bool kill_enable)
{
    if (kill_enable || !relay_enable)
    {
        Relay_AllOff();  // Kill 또는 비활성 상태에서 전원을 차단한다.
    }
    else
    {
        Relay_AllOn();   // 정상 허가 상태에서 전원을 공급한다.
    }
}

/* 한 릴레이의 최근 출력 상태를 반환한다. */
bool Relay_IsOn(Relay_Channel_t channel)
{
    if (!Relay_IsValidChannel(channel))
    {
        return false;
    }

    return (relay_state_mask & (uint8_t)(1U << (uint32_t)channel)) != 0U;  // 상태 비트를 확인한다.
}

/* 여섯 릴레이의 최근 출력 상태를 반환한다. */
uint8_t Relay_GetStateMask(void)
{
    return relay_state_mask;  // Telemetry용 상태 비트를 반환한다.
}
