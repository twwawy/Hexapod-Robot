#ifndef LOW_CONTROL_RELAY_H
#define LOW_CONTROL_RELAY_H

#include <stdbool.h>
#include <stdint.h>

typedef enum
{
    RELAY_INA1 = 0,       // 오른쪽 A 릴레이를 선택한다.
    RELAY_INB1,           // 오른쪽 B 릴레이를 선택한다.
    RELAY_INC1,           // 오른쪽 C 릴레이를 선택한다.
    RELAY_INA2,           // 왼쪽 A 릴레이를 선택한다.
    RELAY_INB2,           // 왼쪽 B 릴레이를 선택한다.
    RELAY_INC2,           // 왼쪽 C 릴레이를 선택한다.
    RELAY_CHANNEL_COUNT   // 전체 릴레이 수를 나타낸다.
} Relay_Channel_t;

void Relay_Init(void);                                      // 모든 릴레이를 OFF로 초기화한다.
void Relay_Set(Relay_Channel_t channel, bool on);           // 한 릴레이 상태를 설정한다.
void Relay_On(Relay_Channel_t channel);                     // 한 릴레이를 켠다.
void Relay_Off(Relay_Channel_t channel);                    // 한 릴레이를 끈다.
void Relay_AllOn(void);                                     // 모든 릴레이를 켠다.
void Relay_AllOff(void);                                    // 모든 릴레이를 끈다.
void Relay_ApplySafety(bool relay_enable, bool kill_enable);// Kill을 반영한 출력을 적용한다.
bool Relay_IsOn(Relay_Channel_t channel);                   // 한 릴레이 상태를 반환한다.
uint8_t Relay_GetStateMask(void);                           // 여섯 릴레이 상태를 비트로 반환한다.

#endif
