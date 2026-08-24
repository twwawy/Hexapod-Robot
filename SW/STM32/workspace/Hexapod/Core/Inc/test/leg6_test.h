#ifndef LEG6_TEST_H
#define LEG6_TEST_H

#include "stm32f4xx_hal.h"

#include <stdint.h>

#define LEG6_TEST_JOINT_COUNT       3U
#define LEG6_TEST_NEUTRAL_US        1500U
#define LEG6_TEST_MIN_US            1400U
#define LEG6_TEST_MAX_US            1600U
#define LEG6_TEST_STEP_US           5U
#define LEG6_TEST_STEP_INTERVAL_MS  20U
#define LEG6_TEST_HOLD_MS           700U

typedef enum
{
    LEG6_TEST_IDLE = 0,  // 정지 상태를 나타낸다.
    LEG6_TEST_RUNNING,   // 실행 중 상태를 나타낸다.
    LEG6_TEST_COMPLETE,  // 정상 완료 상태를 나타낸다.
    LEG6_TEST_ERROR      // PWM 오류 상태를 나타낸다.
} Leg6Test_State_t;

typedef struct
{
    TIM_HandleTypeDef *timer[LEG6_TEST_JOINT_COUNT];  // 6번 다리 타이머를 저장한다.
    uint32_t channel[LEG6_TEST_JOINT_COUNT];          // 6번 다리 채널을 저장한다.
    uint16_t pulse_us[LEG6_TEST_JOINT_COUNT];         // 최근 Pulse를 저장한다.
    uint8_t active_joint;                             // 현재 시험 관절을 저장한다.
    uint8_t target_index;                             // 현재 목표 순번을 저장한다.
    uint8_t holding;                                  // 목표 유지 상태를 저장한다.
    uint32_t last_step_ms;                            // 마지막 Ramp 시각을 저장한다.
    uint32_t hold_start_ms;                           // 목표 유지 시작 시각을 저장한다.
    Leg6Test_State_t state;                           // 현재 시험 상태를 저장한다.
} Leg6Test_Handle_t;

HAL_StatusTypeDef Leg6Test_Start(Leg6Test_Handle_t *handle,
                                 TIM_HandleTypeDef *tim1,
                                 TIM_HandleTypeDef *tim3,
                                 TIM_HandleTypeDef *tim4);  // 6번 다리 PWM 시험을 시작한다.
void Leg6Test_Process(Leg6Test_Handle_t *handle);  // 한 관절씩 Ramp 시험을 진행한다.
void Leg6Test_Stop(Leg6Test_Handle_t *handle);     // 세 관절을 중립으로 두고 정지한다.

#endif
