#include "module/relay.h"

#include "main.h"

typedef struct
{
    GPIO_TypeDef *port;
    uint16_t pin;
} Relay_Pin_t;

static const Relay_Pin_t relay_pins[RELAY_CHANNEL_COUNT] =
{
    {INA_GPIO_Port, INA_Pin},
    {INB_GPIO_Port, INB_Pin},
    {INC_GPIO_Port, INC_Pin}
};

static bool Relay_IsValidChannel(Relay_Channel_t channel)
{
    return ((unsigned int)channel < (unsigned int)RELAY_CHANNEL_COUNT);
}

void Relay_Init(void)
{
    Relay_AllOff();
}

void Relay_Set(Relay_Channel_t channel, bool on)
{
    if (!Relay_IsValidChannel(channel))
    {
        return;
    }

    HAL_GPIO_WritePin(relay_pins[channel].port,
                      relay_pins[channel].pin,
                      on ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

void Relay_On(Relay_Channel_t channel)
{
    Relay_Set(channel, true);
}

void Relay_Off(Relay_Channel_t channel)
{
    Relay_Set(channel, false);
}

void Relay_AllOn(void)
{
    Relay_On(RELAY_INA);
    Relay_On(RELAY_INB);
    Relay_On(RELAY_INC);
}

void Relay_AllOff(void)
{
    Relay_Off(RELAY_INA);
    Relay_Off(RELAY_INB);
    Relay_Off(RELAY_INC);
}

bool Relay_IsOn(Relay_Channel_t channel)
{
    if (!Relay_IsValidChannel(channel))
    {
        return false;
    }

    return (HAL_GPIO_ReadPin(relay_pins[channel].port,
                            relay_pins[channel].pin) == GPIO_PIN_SET);
}
