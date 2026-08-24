#ifndef SERVO_RELAY_CALIBRATION_TEST_H
#define SERVO_RELAY_CALIBRATION_TEST_H

#include "low_control/relay.h"
#include "low_control/servo_pwm.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    ServoPwm_Calibration_t servo[ROBOT_JOINT_COUNT];  // 확인한 서보 보정값을 저장한다.
    uint8_t relay_for_leg[ROBOT_LEG_COUNT];           // 다리별 실제 릴레이 번호를 저장한다.
    bool relay_mapped[ROBOT_LEG_COUNT];               // 릴레이 대응 확인 여부를 저장한다.
} ServoRelayCalibrationTest_t;

void ServoRelayCalibrationTest_Init(ServoRelayCalibrationTest_t *test);  // 임시 서보값과 미확인 릴레이표를 준비한다.
bool ServoRelayCalibrationTest_RecordServo(ServoRelayCalibrationTest_t *test,
                                           uint8_t joint,
                                           uint16_t minimum_us,
                                           uint16_t neutral_us,
                                           uint16_t maximum_us,
                                           int8_t direction);  // 한 서보의 실측값을 기록한다.
bool ServoRelayCalibrationTest_RecordRelay(ServoRelayCalibrationTest_t *test,
                                           uint8_t leg,
                                           Relay_Channel_t relay);  // 한 다리의 릴레이 대응을 기록한다.
bool ServoRelayCalibrationTest_ApplyServos(const ServoRelayCalibrationTest_t *test,
                                           ServoPwm_Handle_t *servo);  // 완료된 서보값을 출력 모듈에 적용한다.

#endif
