#ifndef MODULE_RELAY_H
#define MODULE_RELAY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

typedef enum
{
    RELAY_INA1 = 0,       // 오른쪽 A 릴레이를 선택한다.
    RELAY_INB1,           // 오른쪽 B 릴레이를 선택한다.
    RELAY_INC1,           // 오른쪽 C 릴레이를 선택한다.
    RELAY_INA2,           // 왼쪽 A 릴레이를 선택한다.
    RELAY_INB2,           // 왼쪽 B 릴레이를 선택한다.
    RELAY_INC2,           // 왼쪽 C 릴레이를 선택한다.
    RELAY_CHANNEL_COUNT   // 전체 릴레이 채널 수를 나타낸다.
} Relay_Channel_t;

void Relay_Init(void);                                  // 모든 릴레이를 OFF로 초기화한다.

void Relay_Set(Relay_Channel_t channel, bool on);       // 지정한 릴레이 상태를 변경한다.

void Relay_On(Relay_Channel_t channel);                 // 지정한 릴레이를 ON으로 설정한다.

void Relay_Off(Relay_Channel_t channel);                // 지정한 릴레이를 OFF로 설정한다.

void Relay_AllOn(void);                                 // 모든 릴레이를 ON으로 설정한다.

void Relay_AllOff(void);                                // 모든 릴레이를 OFF로 설정한다.

bool Relay_IsOn(Relay_Channel_t channel);               // 지정한 릴레이의 ON 상태를 확인한다.

#ifdef __cplusplus
}
#endif

#endif /* MODULE_RELAY_H */
