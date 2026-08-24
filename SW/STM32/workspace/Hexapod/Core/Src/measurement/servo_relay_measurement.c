#include "measurement/servo_relay_measurement.h"

#include "common/robot_config.h"
#include "measurement/measurement_debug.h"

#include <stddef.h>
#include <string.h>

/* 릴레이를 차단하고 18개 PWM을 중립에서 시작한다. */
bool ServoRelayMeasurement_Init(ServoRelayMeasurement_t *measurement,
                                ServoPwm_Handle_t *servo_output)
{
    uint32_t joint;  // 초기화할 서보 번호를 저장한다.
    uint32_t leg;    // 초기화할 릴레이 매핑 번호를 저장한다.

    if ((measurement == NULL) || (servo_output == NULL))
    {
        return false;
    }

    memset(measurement, 0, sizeof(*measurement));  // 이전 실측값을 제거한다.
    measurement->servo_output = servo_output;      // PWM 출력 상태를 연결한다.
    Relay_Init();                                  // 모터 전원을 모두 차단한다.
    g_measurement_debug.active_servo_joint = UINT8_MAX;  // 선택된 서보가 없음을 표시한다.
    g_measurement_debug.active_servo_pulse_us = 0U;      // 최근 Pulse를 초기화한다.
    g_measurement_debug.relay_state_mask = 0U;           // 릴레이 상태를 초기화한다.
    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        g_measurement_debug.calibration.servo[joint].calibrated = false;  // 서보별 완료를 초기화한다.
    }
    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        g_measurement_debug.calibration.relay_mapped[leg] = false;  // 릴레이별 완료를 초기화한다.
    }

    if (!servo_output->started && (ServoPwm_Start(servo_output) != HAL_OK))
    {
        return false;
    }

    measurement->output_started = true;  // 중립 PWM 시작을 표시한다.
    return true;
}

/* 선택한 서보 한 개에 제한된 실측 Pulse를 직접 출력한다. */
bool ServoRelayMeasurement_WritePulse(ServoRelayMeasurement_t *measurement,
                                      uint8_t joint,
                                      uint16_t pulse_us)
{
    if ((measurement == NULL) || (measurement->servo_output == NULL) ||
        !measurement->output_started || (joint >= ROBOT_JOINT_COUNT) ||
        (pulse_us < ROBOT_SERVO_MIN_US) || (pulse_us > ROBOT_SERVO_MAX_US))
    {
        return false;
    }

    __HAL_TIM_SET_COMPARE(measurement->servo_output->timer[joint],
                          measurement->servo_output->channel[joint],
                          pulse_us);                                        // 선택 관절의 Compare만 갱신한다.
    measurement->servo_output->pulse_us[joint] = pulse_us;                  // 최근 실측 Pulse를 기록한다.
    g_measurement_debug.active_servo_joint = joint;                          // 디버거 선택 서보를 갱신한다.
    g_measurement_debug.active_servo_pulse_us = pulse_us;                    // 디버거 최근 Pulse를 갱신한다.
    return true;
}

/* 사용자가 선택한 한 릴레이만 켜거나 끈다. */
bool ServoRelayMeasurement_SetRelay(ServoRelayMeasurement_t *measurement,
                                    Relay_Channel_t relay,
                                    bool on)
{
    if ((measurement == NULL) || !measurement->output_started ||
        ((uint32_t)relay >= (uint32_t)RELAY_CHANNEL_COUNT))
    {
        return false;
    }

    Relay_Set(relay, on);  // 명시적으로 선택한 전원만 변경한다.
    g_measurement_debug.relay_state_mask = Relay_GetStateMask();  // 디버거 릴레이 상태를 갱신한다.
    return Relay_IsOn(relay) == on;
}

/* 한 서보의 Pulse 범위와 대응 관절각·방향을 기록한다. */
bool ServoRelayMeasurement_RecordServo(ServoRelayMeasurement_t *measurement,
                                       uint8_t joint,
                                       uint16_t minimum_us,
                                       uint16_t neutral_us,
                                       uint16_t maximum_us,
                                       float angle_min_rad,
                                       float zero_angle_rad,
                                       float angle_max_rad,
                                       int8_t direction)
{
    ServoPwm_Calibration_t *calibration;  // 기록할 서보 보정값을 참조한다.

    if ((measurement == NULL) || (joint >= ROBOT_JOINT_COUNT) ||
        (minimum_us >= neutral_us) || (neutral_us >= maximum_us) ||
        (angle_min_rad >= zero_angle_rad) || (zero_angle_rad >= angle_max_rad) ||
        ((direction != 1) && (direction != -1)))
    {
        return false;
    }

    calibration = &measurement->servo[joint];                     // 선택 서보 표를 연다.
    calibration->minimum_us = minimum_us;                          // 실측 최소 Pulse를 저장한다.
    calibration->neutral_us = neutral_us;                          // 실측 중립 Pulse를 저장한다.
    calibration->maximum_us = maximum_us;                          // 실측 최대 Pulse를 저장한다.
    calibration->zero_angle_rad = zero_angle_rad;                  // 중립 Pulse의 관절각을 저장한다.
    calibration->pulse_per_rad = (float)(maximum_us - minimum_us) /
                                 (angle_max_rad - angle_min_rad);   // 실측 각도당 Pulse 비율을 계산한다.
    calibration->direction = direction;                            // 확인한 회전 방향을 저장한다.
    calibration->calibrated = true;                                // 실측 완료를 표시한다.
    g_measurement_debug.calibration.servo[joint] = *calibration;    // 디버거 최종 서보값을 갱신한다.
    return true;
}

/* 한 릴레이만 켜서 확인한 다리 대응을 기록한다. */
bool ServoRelayMeasurement_RecordRelay(ServoRelayMeasurement_t *measurement,
                                       uint8_t leg,
                                       Relay_Channel_t relay)
{
    if ((measurement == NULL) || (leg >= ROBOT_LEG_COUNT) ||
        ((uint32_t)relay >= (uint32_t)RELAY_CHANNEL_COUNT))
    {
        return false;
    }

    measurement->relay_for_leg[leg] = relay;  // 관측한 릴레이 번호를 저장한다.
    measurement->relay_mapped[leg] = true;    // 대응 확인을 표시한다.
    g_measurement_debug.calibration.relay_for_leg[leg] = relay;  // 디버거 릴레이 대응을 갱신한다.
    g_measurement_debug.calibration.relay_mapped[leg] = true;    // 디버거 대응 완료를 표시한다.
    return true;
}

/* 실측 종료 시 모든 릴레이와 PWM을 정지한다. */
void ServoRelayMeasurement_Stop(ServoRelayMeasurement_t *measurement)
{
    Relay_AllOff();  // 모든 모터 전원을 먼저 차단한다.
    g_measurement_debug.relay_state_mask = 0U;  // 디버거 릴레이 상태를 OFF로 갱신한다.
    if ((measurement != NULL) && (measurement->servo_output != NULL))
    {
        ServoPwm_Stop(measurement->servo_output);  // 모든 PWM 채널을 정지한다.
        measurement->output_started = false;       // 출력 종료를 표시한다.
    }
}
