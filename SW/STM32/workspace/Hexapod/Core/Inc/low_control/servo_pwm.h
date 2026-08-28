#ifndef SERVO_PWM_H
#define SERVO_PWM_H

#include "common/robot_types.h"
#include "stm32f4xx_hal.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    TIM_HandleTypeDef *tim1;  // TIM1 Handle을 저장한다.
    TIM_HandleTypeDef *tim2;  // TIM2 Handle을 저장한다.
    TIM_HandleTypeDef *tim3;  // TIM3 Handle을 저장한다.
    TIM_HandleTypeDef *tim4;  // TIM4 Handle을 저장한다.
    TIM_HandleTypeDef *tim5;  // TIM5 Handle을 저장한다.
    TIM_HandleTypeDef *tim8;  // TIM8 Handle을 저장한다.
} ServoPwm_TimerBank_t;

typedef struct
{
    uint16_t neutral_us;     // 서보 중립 Pulse를 저장한다.
    uint16_t minimum_us;     // 서보 최소 Pulse를 저장한다.
    uint16_t maximum_us;     // 서보 최대 Pulse를 저장한다.
    float zero_angle_rad;    // 중립 Pulse의 관절각을 저장한다.
    float pulse_per_rad;     // rad당 Pulse 변화량을 저장한다.
    int8_t direction;        // 서보 회전 방향을 저장한다.
    bool calibrated;        // 실측 완료 여부를 저장한다.
} ServoPwm_Calibration_t;

typedef struct
{
    TIM_HandleTypeDef *timer[ROBOT_JOINT_COUNT];       // 관절별 타이머를 저장한다.
    uint32_t channel[ROBOT_JOINT_COUNT];               // 관절별 타이머 채널을 저장한다.
    ServoPwm_Calibration_t table[ROBOT_JOINT_COUNT];   // 관절별 서보 보정값을 저장한다.
    float previous_angle_rad[ROBOT_JOINT_COUNT];       // Rate Limit 이전값을 저장한다.
    uint16_t pulse_us[ROBOT_JOINT_COUNT];              // 최근 출력 Pulse를 저장한다.
    bool seeded;                                       // 이전 관절각 초기화 여부를 저장한다.
    bool started;                                      // PWM 시작 여부를 저장한다.
} ServoPwm_Handle_t;

void ServoPwm_Init(ServoPwm_Handle_t *handle,
                   const ServoPwm_TimerBank_t *timers);  // 18개 PWM 채널을 배치한다.

bool ServoPwm_SetCalibration(ServoPwm_Handle_t *handle,
                             uint8_t joint,
                             const ServoPwm_Calibration_t *calibration);  // 한 서보 보정값을 갱신한다.

bool ServoPwm_CalculatePulse(const ServoPwm_Calibration_t *calibration,
                             float angle_rad,
                             uint16_t *pulse_us);  // 보정된 관절각을 PWM Pulse로 계산한다.

HAL_StatusTypeDef ServoPwm_Start(ServoPwm_Handle_t *handle);  // 모든 PWM을 중립에서 시작한다.

HAL_StatusTypeDef ServoPwm_StartAngles(ServoPwm_Handle_t *handle,
                                       const float angle_rad[ROBOT_JOINT_COUNT]);  // 주어진 관절각에서 PWM을 시작한다.

void ServoPwm_SeedAngles(ServoPwm_Handle_t *handle,
                         const float angle_rad[ROBOT_JOINT_COUNT]);  // 현재 측정각으로 Rate Limit를 초기화한다.

bool ServoPwm_WriteAngles(ServoPwm_Handle_t *handle,
                          const float target_rad[ROBOT_JOINT_COUNT]);  // 제한된 18개 관절 명령을 출력한다.

void ServoPwm_Stop(ServoPwm_Handle_t *handle);  // 모든 PWM 출력을 정지한다.

#endif
