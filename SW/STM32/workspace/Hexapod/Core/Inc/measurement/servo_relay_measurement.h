#ifndef SERVO_RELAY_MEASUREMENT_H
#define SERVO_RELAY_MEASUREMENT_H

#include "low_control/relay.h"
#include "low_control/servo_pwm.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    ServoPwm_Handle_t *servo_output;                      // 실측할 PWM 출력을 참조한다.
    ServoPwm_Calibration_t servo[ROBOT_JOINT_COUNT];      // 확인한 서보 보정값을 저장한다.
    Relay_Channel_t relay_for_leg[ROBOT_LEG_COUNT];       // 다리별 실제 릴레이를 저장한다.
    bool relay_mapped[ROBOT_LEG_COUNT];                   // 릴레이 대응 확인 여부를 저장한다.
    bool output_started;                                  // PWM 출력 시작 여부를 저장한다.
} ServoRelayMeasurement_t;

bool ServoRelayMeasurement_Init(ServoRelayMeasurement_t *measurement,
                                ServoPwm_Handle_t *servo_output);  // 릴레이 OFF와 중립 PWM을 준비한다.
bool ServoRelayMeasurement_WritePulse(ServoRelayMeasurement_t *measurement,
                                      uint8_t joint,
                                      uint16_t pulse_us);  // 선택한 한 서보에 실측 Pulse를 출력한다.
bool ServoRelayMeasurement_SetRelay(ServoRelayMeasurement_t *measurement,
                                    Relay_Channel_t relay,
                                    bool on);  // 선택한 한 릴레이를 설정한다.
bool ServoRelayMeasurement_RecordServo(ServoRelayMeasurement_t *measurement,
                                       uint8_t joint,
                                       uint16_t minimum_us,
                                       uint16_t neutral_us,
                                       uint16_t maximum_us,
                                       float angle_min_rad,
                                       float zero_angle_rad,
                                       float angle_max_rad,
                                       int8_t direction);  // 한 서보의 실측값을 기록한다.
bool ServoRelayMeasurement_RecordRelay(ServoRelayMeasurement_t *measurement,
                                       uint8_t leg,
                                       Relay_Channel_t relay);  // 한 다리의 릴레이 대응을 기록한다.
void ServoRelayMeasurement_Stop(ServoRelayMeasurement_t *measurement);  // 릴레이와 PWM을 안전하게 정지한다.

#endif
