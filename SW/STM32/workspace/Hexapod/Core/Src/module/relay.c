#include "module/relay.h"

#include "main.h"

#include <stdint.h>

typedef struct
{
    GPIO_TypeDef *port;   // 릴레이 GPIO 포트를 저장한다.
    uint16_t pin;         // 릴레이 GPIO 핀을 저장한다.
} Relay_Pin_t;

static const Relay_Pin_t relay_pins[RELAY_CHANNEL_COUNT] =   // 채널별 실제 GPIO를 연결한다.
{
    {INA1_GPIO_Port, INA1_Pin},   // 오른쪽 A 릴레이를 연결한다.
    {INB1_GPIO_Port, INB1_Pin},   // 오른쪽 B 릴레이를 연결한다.
    {INC1_GPIO_Port, INC1_Pin},   // 오른쪽 C 릴레이를 연결한다.
    {INA2_GPIO_Port, INA2_Pin},   // 왼쪽 A 릴레이를 연결한다.
    {INB2_GPIO_Port, INB2_Pin},   // 왼쪽 B 릴레이를 연결한다.
    {INC2_GPIO_Port, INC2_Pin}    // 왼쪽 C 릴레이를 연결한다.
};

/* 지정한 릴레이 채널의 유효성을 검사한다. */
static bool Relay_IsValidChannel(Relay_Channel_t channel)
{
    return ((uint32_t)channel < (uint32_t)RELAY_CHANNEL_COUNT);   // 배열 접근 가능 여부를 반환한다.
}

/* 모든 릴레이를 안전한 OFF 상태로 초기화한다. */
void Relay_Init(void)
{
    Relay_AllOff();   // 초기 서보 전원 공급을 차단한다.
}

/* 지정한 릴레이 채널의 출력 상태를 변경한다. */
void Relay_Set(Relay_Channel_t channel, bool on)
{
    GPIO_PinState state;   // 요청한 GPIO 출력 상태를 저장한다.

    if (!Relay_IsValidChannel(channel))
    {
        return;
    }

    state = on ? GPIO_PIN_SET : GPIO_PIN_RESET;                   // 활성 HIGH 상태로 변환한다.
    HAL_GPIO_WritePin(relay_pins[channel].port,
                      relay_pins[channel].pin,
                      state);                                    // 선택한 릴레이에 상태를 출력한다.
}

/* 지정한 릴레이 채널을 ON으로 설정한다. */
void Relay_On(Relay_Channel_t channel)
{
    Relay_Set(channel, true);   // 선택한 릴레이 전원을 공급한다.
}

/* 지정한 릴레이 채널을 OFF로 설정한다. */
void Relay_Off(Relay_Channel_t channel)
{
    Relay_Set(channel, false);   // 선택한 릴레이 전원을 차단한다.
}

/* 모든 릴레이 채널을 ON으로 설정한다. */
void Relay_AllOn(void)
{
    for (uint32_t index = 0U; index < (uint32_t)RELAY_CHANNEL_COUNT; ++index)   // 전체 채널을 순회한다.
    {
        Relay_On((Relay_Channel_t)index);   // 현재 릴레이 전원을 공급한다.
    }
}

/* 모든 릴레이 채널을 OFF로 설정한다. */
void Relay_AllOff(void)
{
    for (uint32_t index = 0U; index < (uint32_t)RELAY_CHANNEL_COUNT; ++index)   // 전체 채널을 순회한다.
    {
        Relay_Off((Relay_Channel_t)index);   // 현재 릴레이 전원을 차단한다.
    }
}

/* 지정한 릴레이 채널의 현재 출력 상태를 확인한다. */
bool Relay_IsOn(Relay_Channel_t channel)
{
    GPIO_PinState state;   // 현재 GPIO 출력 상태를 저장한다.

    if (!Relay_IsValidChannel(channel))
    {
        return false;
    }

    state = HAL_GPIO_ReadPin(relay_pins[channel].port,
                             relay_pins[channel].pin);   // 선택한 릴레이 출력을 읽는다.

    return (state == GPIO_PIN_SET);                      // 활성 HIGH 여부를 반환한다.
}
