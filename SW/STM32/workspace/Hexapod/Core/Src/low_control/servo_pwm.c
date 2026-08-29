#include "low_control/servo_pwm.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

/* 실수 값을 지정한 범위로 제한한다. */
static float ServoPwm_Clamp(float value, float minimum, float maximum)
{
    if (value < minimum)
    {
        return minimum;
    }
    if (value > maximum)
    {
        return maximum;
    }
    return value;
}

/* Pulse 값을 지정한 범위로 제한한다. */
static uint16_t ServoPwm_ClampPulse(int32_t value,
                                    uint16_t minimum,
                                    uint16_t maximum)
{
    if (value < (int32_t)minimum)
    {
        return minimum;
    }
    if (value > (int32_t)maximum)
    {
        return maximum;
    }
    return (uint16_t)value;
}

/* 타이머와 채널 한 쌍을 관절에 연결한다. */
static void ServoPwm_Map(ServoPwm_Handle_t *handle,
                         uint8_t joint,
                         TIM_HandleTypeDef *timer,
                         uint32_t channel)
{
    handle->timer[joint] = timer;      // 관절 타이머를 저장한다.
    handle->channel[joint] = channel;  // 관절 채널을 저장한다.
}

/* CubeMX의 18개 PWM 채널과 기본 보정값을 준비한다. */
void ServoPwm_Init(ServoPwm_Handle_t *handle,
                   const ServoPwm_TimerBank_t *timers)
{
    uint32_t joint;   // 초기화할 관절 번호를 저장한다.
    const float default_pulse_per_rad = 2000.0f / (270.0f * ROBOT_DEG_TO_RAD_F);  // 270도 Pulse 비율을 계산한다.

    if ((handle == NULL) || (timers == NULL))
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));  // 이전 PWM 상태를 제거한다.

    ServoPwm_Map(handle, 0U, timers->tim1, TIM_CHANNEL_1);   // 1_1 채널을 연결한다.
    ServoPwm_Map(handle, 1U, timers->tim1, TIM_CHANNEL_2);   // 1_2 채널을 연결한다.
    ServoPwm_Map(handle, 2U, timers->tim1, TIM_CHANNEL_3);   // 1_3 채널을 연결한다.
    ServoPwm_Map(handle, 3U, timers->tim2, TIM_CHANNEL_1);   // 2_1 채널을 연결한다.
    ServoPwm_Map(handle, 4U, timers->tim2, TIM_CHANNEL_2);   // 2_2 채널을 연결한다.
    ServoPwm_Map(handle, 5U, timers->tim2, TIM_CHANNEL_3);   // 2_3 채널을 연결한다.
    ServoPwm_Map(handle, 6U, timers->tim3, TIM_CHANNEL_1);   // 3_1 채널을 연결한다.
    ServoPwm_Map(handle, 7U, timers->tim3, TIM_CHANNEL_2);   // 3_2 채널을 연결한다.
    ServoPwm_Map(handle, 8U, timers->tim3, TIM_CHANNEL_3);   // 3_3 채널을 연결한다.
    ServoPwm_Map(handle, 9U, timers->tim4, TIM_CHANNEL_1);   // 4_1 채널을 연결한다.
    ServoPwm_Map(handle, 10U, timers->tim4, TIM_CHANNEL_3);  // 4_2를 PB8의 TIM4 CH3에 연결한다.
    ServoPwm_Map(handle, 11U, timers->tim4, TIM_CHANNEL_2);  // 4_3을 PB7의 TIM4 CH2에 연결한다.
    ServoPwm_Map(handle, 12U, timers->tim5, TIM_CHANNEL_1);  // 5_1 채널을 연결한다.
    ServoPwm_Map(handle, 13U, timers->tim5, TIM_CHANNEL_2);  // 5_2 채널을 연결한다.
    ServoPwm_Map(handle, 14U, timers->tim8, TIM_CHANNEL_3);  // 5_3 채널을 연결한다.
    ServoPwm_Map(handle, 15U, timers->tim1, TIM_CHANNEL_4);  // 6_1 채널을 연결한다.
    ServoPwm_Map(handle, 16U, timers->tim3, TIM_CHANNEL_4);  // 6_2 채널을 연결한다.
    ServoPwm_Map(handle, 17U, timers->tim4, TIM_CHANNEL_4);  // 6_3 채널을 연결한다.

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        handle->table[joint].neutral_us = ROBOT_SERVO_NEUTRAL_US;  // 임시 중립 Pulse를 넣는다.
        handle->table[joint].minimum_us = ROBOT_SERVO_MIN_US;      // 임시 최소 Pulse를 넣는다.
        handle->table[joint].maximum_us = ROBOT_SERVO_MAX_US;      // 임시 최대 Pulse를 넣는다.
        handle->table[joint].zero_angle_rad = 0.0f;                // 임시 영점각을 넣는다.
        handle->table[joint].pulse_per_rad = default_pulse_per_rad;// 기본 Pulse 비율을 넣는다.
        handle->table[joint].direction = 1;                        // 임시 정방향을 넣는다.
        handle->table[joint].calibrated = false;                   // 실측 전 상태로 표시한다.
        handle->pulse_us[joint] = ROBOT_SERVO_NEUTRAL_US;          // 초기 출력값을 저장한다.
    }
}

/* 한 서보의 실측 보정값을 설정한다. */
bool ServoPwm_SetCalibration(ServoPwm_Handle_t *handle,
                             uint8_t joint,
                             const ServoPwm_Calibration_t *calibration)
{
    if ((handle == NULL) || (calibration == NULL) ||
        (joint >= ROBOT_JOINT_COUNT) ||
        (calibration->minimum_us >= calibration->neutral_us) ||
        (calibration->neutral_us >= calibration->maximum_us) ||
        (calibration->pulse_per_rad <= 0.0f))
    {
        return false;
    }

    handle->table[joint] = *calibration;  // 선택한 서보 테이블을 갱신한다.
    return true;
}

/* 한 서보의 보정값으로 관절각을 제한된 PWM Pulse로 계산한다. */
bool ServoPwm_CalculatePulse(const ServoPwm_Calibration_t *calibration,
                             float angle_rad,
                             uint16_t *pulse_us)
{
    int32_t pulse;  // 변환한 Pulse를 저장한다.

    if ((calibration == NULL) || (pulse_us == NULL) ||
        !calibration->calibrated ||
        (calibration->minimum_us >= calibration->neutral_us) ||
        (calibration->neutral_us >= calibration->maximum_us) ||
        (calibration->pulse_per_rad <= 0.0f) ||
        ((calibration->direction != 1) && (calibration->direction != -1)))
    {
        return false;
    }

    pulse = (int32_t)lroundf((float)calibration->neutral_us +
                             (float)calibration->direction *
                             (angle_rad - calibration->zero_angle_rad) *
                             calibration->pulse_per_rad);                    // 관절각을 Pulse로 변환한다.
    *pulse_us = ServoPwm_ClampPulse(pulse, calibration->minimum_us,
                                    calibration->maximum_us);                // 실측 Pulse 범위로 제한한다.
    return true;
}

/* 모든 서보 PWM을 중립 Pulse에서 시작한다. */
HAL_StatusTypeDef ServoPwm_Start(ServoPwm_Handle_t *handle)
{
    uint32_t joint;   // 시작할 관절 번호를 저장한다.

    if (handle == NULL)
    {
        return HAL_ERROR;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        if (handle->timer[joint] == NULL)
        {
            ServoPwm_Stop(handle);  // 불완전한 타이머 배치를 정리한다.
            return HAL_ERROR;
        }

        __HAL_TIM_SET_COMPARE(handle->timer[joint],
                              handle->channel[joint],
                              handle->table[joint].neutral_us);  // 중립 Pulse를 먼저 기록한다.

        if (HAL_TIM_PWM_Start(handle->timer[joint], handle->channel[joint]) != HAL_OK)
        {
            ServoPwm_Stop(handle);  // 시작된 PWM을 모두 정리한다.
            return HAL_ERROR;
        }

        handle->pulse_us[joint] = handle->table[joint].neutral_us;  // 최근 Pulse를 갱신한다.
    }

    handle->started = true;  // 전체 PWM 시작을 표시한다.
    return HAL_OK;
}

/* 주어진 관절각 Pulse를 먼저 기록하고 PWM을 시작한다. */
HAL_StatusTypeDef ServoPwm_StartAngles(ServoPwm_Handle_t *handle,
                                       const float angle_rad[ROBOT_JOINT_COUNT])
{
    uint16_t pulse_us[ROBOT_JOINT_COUNT];  // 관절별 시작 Pulse를 저장한다.
    uint32_t joint;                        // 시작할 관절 번호를 저장한다.

    if ((handle == NULL) || (angle_rad == NULL) || handle->started)
    {
        return HAL_ERROR;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        const float limited = ServoPwm_Clamp(angle_rad[joint],
                                             ROBOT_JOINT_MIN_RAD,
                                             ROBOT_JOINT_MAX_RAD);  // 시작각을 관절 범위로 제한한다.

        if ((handle->timer[joint] == NULL) ||
            !ServoPwm_CalculatePulse(&handle->table[joint],
                                     limited,
                                     &pulse_us[joint]))
        {
            return HAL_ERROR;
        }
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        __HAL_TIM_SET_COMPARE(handle->timer[joint],
                              handle->channel[joint],
                              pulse_us[joint]);              // PWM 시작 전에 실측각 Pulse를 기록한다.
        handle->pulse_us[joint] = pulse_us[joint];            // 최근 Pulse를 갱신한다.
        handle->previous_angle_rad[joint] = ServoPwm_Clamp(
            angle_rad[joint],
            ROBOT_JOINT_MIN_RAD,
            ROBOT_JOINT_MAX_RAD);                             // 초기 PWM 명령각을 맞춘다.
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        if (HAL_TIM_PWM_Start(handle->timer[joint], handle->channel[joint]) != HAL_OK)
        {
            ServoPwm_Stop(handle);  // 시작된 PWM을 모두 정리한다.
            return HAL_ERROR;
        }
    }

    handle->seeded = true;   // PWM 명령각 초기화를 표시한다.
    handle->started = true;  // 전체 PWM 시작을 표시한다.
    return HAL_OK;
}

/* 현재 측정 관절각으로 PWM 명령값을 초기화한다. */
void ServoPwm_SeedAngles(ServoPwm_Handle_t *handle,
                         const float angle_rad[ROBOT_JOINT_COUNT])
{
    uint32_t joint;   // 초기화할 관절 번호를 저장한다.

    if ((handle == NULL) || (angle_rad == NULL))
    {
        return;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        handle->previous_angle_rad[joint] =
            ServoPwm_Clamp(angle_rad[joint], ROBOT_JOINT_MIN_RAD, ROBOT_JOINT_MAX_RAD);  // 현재 관절각을 안전 범위로 저장한다.
    }

    handle->seeded = true;  // PWM 명령각 초기화를 표시한다.
}

/* 관절 범위를 제한하여 18개 PWM으로 출력한다. */
bool ServoPwm_WriteAngles(ServoPwm_Handle_t *handle,
                          const float target_rad[ROBOT_JOINT_COUNT])
{
    uint32_t joint;   // 출력할 관절 번호를 저장한다.

    if ((handle == NULL) || (target_rad == NULL) || !handle->started)
    {
        return false;
    }

    if (!handle->seeded)
    {
        ServoPwm_SeedAngles(handle, target_rad);  // 첫 유효 PWM 명령값을 초기화한다.
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        const ServoPwm_Calibration_t *calibration = &handle->table[joint];  // 현재 서보 보정값을 선택한다.
        const float target = ServoPwm_Clamp(target_rad[joint],
                                            ROBOT_JOINT_MIN_RAD,
                                            ROBOT_JOINT_MAX_RAD);           // 목표 관절각을 제한한다.
        uint16_t pulse;                                                     // 변환한 Pulse를 저장한다.

        if (!ServoPwm_CalculatePulse(calibration, target, &pulse))
        {
            return false;
        }
        handle->pulse_us[joint] = pulse;                    // 보정된 Pulse를 저장한다.
        handle->previous_angle_rad[joint] = target;         // 최근 PWM 명령각을 저장한다.
        __HAL_TIM_SET_COMPARE(handle->timer[joint],
                              handle->channel[joint],
                              handle->pulse_us[joint]);     // PWM Compare를 갱신한다.
    }

    return true;
}

/* 모든 서보 PWM 출력을 정지한다. */
void ServoPwm_Stop(ServoPwm_Handle_t *handle)
{
    uint32_t joint;   // 정지할 관절 번호를 저장한다.

    if (handle == NULL)
    {
        return;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        if (handle->timer[joint] != NULL)
        {
            (void)HAL_TIM_PWM_Stop(handle->timer[joint], handle->channel[joint]);  // PWM 채널을 정지한다.
            __HAL_TIM_SET_COMPARE(handle->timer[joint],
                                  handle->channel[joint],
                                  handle->table[joint].neutral_us);                // 정지 후 중립 Pulse를 준비한다.
        }
    }

    handle->started = false;  // 전체 PWM 정지를 표시한다.
}
