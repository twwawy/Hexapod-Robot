#include "test/servo_relay_calibration_test.h"

#include <stddef.h>
#include <string.h>

/* 서보 기본값과 미확인 릴레이 대응표를 준비한다. */
void ServoRelayCalibrationTest_Init(ServoRelayCalibrationTest_t *test)
{
    uint32_t joint;  // 초기화할 관절 번호를 저장한다.

    if (test == NULL)
    {
        return;
    }

    memset(test, 0, sizeof(*test));  // 이전 실측값을 제거한다.
    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        test->servo[joint].minimum_us = ROBOT_SERVO_MIN_US;      // 시험 Pulse 최소값을 둔다.
        test->servo[joint].neutral_us = ROBOT_SERVO_NEUTRAL_US;  // 시험 중립값을 둔다.
        test->servo[joint].maximum_us = ROBOT_SERVO_MAX_US;      // 시험 Pulse 최대값을 둔다.
        test->servo[joint].pulse_per_rad = 2000.0f /
            (270.0f * ROBOT_DEG_TO_RAD_F);                       // DS51150-270 기본 비율을 둔다.
        test->servo[joint].direction = 1;                         // 확인 전 방향을 정방향으로 둔다.
    }
}

/* 기구를 분리해 확인한 한 서보의 Pulse 범위와 방향을 기록한다. */
bool ServoRelayCalibrationTest_RecordServo(ServoRelayCalibrationTest_t *test,
                                           uint8_t joint,
                                           uint16_t minimum_us,
                                           uint16_t neutral_us,
                                           uint16_t maximum_us,
                                           int8_t direction)
{
    ServoPwm_Calibration_t *calibration;  // 기록할 서보 보정값을 참조한다.

    if ((test == NULL) || (joint >= ROBOT_JOINT_COUNT) ||
        (minimum_us >= neutral_us) || (neutral_us >= maximum_us) ||
        ((direction != 1) && (direction != -1)))
    {
        return false;
    }

    calibration = &test->servo[joint];                 // 선택 서보 표를 연다.
    calibration->minimum_us = minimum_us;              // 실측 최소 Pulse를 저장한다.
    calibration->neutral_us = neutral_us;              // 실측 중립 Pulse를 저장한다.
    calibration->maximum_us = maximum_us;              // 실측 최대 Pulse를 저장한다.
    calibration->zero_angle_rad = 0.0f;                // 중립 관절각을 0으로 둔다.
    calibration->pulse_per_rad = (float)(maximum_us - minimum_us) /
                                 (270.0f * ROBOT_DEG_TO_RAD_F);  // 실측 범위의 rad 비율을 계산한다.
    calibration->direction = direction;                // 확인한 회전 방향을 저장한다.
    calibration->calibrated = true;                    // 실측 완료를 표시한다.
    return true;
}

/* 한 릴레이만 켜서 확인한 다리 대응을 기록한다. */
bool ServoRelayCalibrationTest_RecordRelay(ServoRelayCalibrationTest_t *test,
                                           uint8_t leg,
                                           Relay_Channel_t relay)
{
    if ((test == NULL) || (leg >= ROBOT_LEG_COUNT) ||
        ((uint32_t)relay >= (uint32_t)RELAY_CHANNEL_COUNT))
    {
        return false;
    }

    test->relay_for_leg[leg] = (uint8_t)relay;  // 관측한 릴레이 번호를 저장한다.
    test->relay_mapped[leg] = true;             // 대응 확인을 표시한다.
    return true;
}

/* 실측 완료된 18개 서보 보정값을 PWM 모듈에 적용한다. */
bool ServoRelayCalibrationTest_ApplyServos(const ServoRelayCalibrationTest_t *test,
                                           ServoPwm_Handle_t *servo)
{
    uint32_t joint;  // 적용할 관절 번호를 저장한다.

    if ((test == NULL) || (servo == NULL))
    {
        return false;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        if (!test->servo[joint].calibrated ||
            !ServoPwm_SetCalibration(servo, (uint8_t)joint, &test->servo[joint]))
        {
            return false;
        }
    }

    return true;
}
