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
    LEG6_TEST_IDLE = 0,
    LEG6_TEST_RUNNING,
    LEG6_TEST_COMPLETE,
    LEG6_TEST_ERROR
} Leg6Test_State_t;

typedef struct
{
    TIM_HandleTypeDef *timer[LEG6_TEST_JOINT_COUNT];
    uint32_t channel[LEG6_TEST_JOINT_COUNT];
    uint16_t pulse_us[LEG6_TEST_JOINT_COUNT];

    uint8_t active_joint;
    uint8_t target_index;
    uint8_t holding;
    uint32_t last_step_ms;
    uint32_t hold_start_ms;
    Leg6Test_State_t state;
} Leg6Test_Handle_t;

/**
 * Start PWM for leg 6 only:
 *   joint 6_1 = TIM1_CH4 / PA11
 *   joint 6_2 = TIM3_CH4 / PB1
 *   joint 6_3 = TIM4_CH4 / PB9
 */
HAL_StatusTypeDef Leg6Test_Start(Leg6Test_Handle_t *handle,
                                 TIM_HandleTypeDef *tim1,
                                 TIM_HandleTypeDef *tim3,
                                 TIM_HandleTypeDef *tim4);

/** Call continuously from the main loop. The test runs once and ends neutral. */
void Leg6Test_Process(Leg6Test_Handle_t *handle);

/** Return all three joints to neutral and stop their PWM outputs. */
void Leg6Test_Stop(Leg6Test_Handle_t *handle);

#endif /* LEG6_TEST_H */
