#include "test/joint_sensor_calibration_test.h"

#include "sensor/mcp3008.h"

#include <stddef.h>
#include <string.h>

/* 관절 ADC 최소·중립·최대 측정 상태를 준비한다. */
void JointSensorCalibrationTest_Init(JointSensorCalibrationTest_t *test)
{
    uint32_t joint;  // 초기화할 관절 번호를 저장한다.

    if (test == NULL)
    {
        return;
    }

    memset(test, 0, sizeof(*test));  // 이전 측정값을 제거한다.
    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        test->minimum[joint] = MCP3008_MAX_RAW_VALUE;  // 새 최소값을 받을 시작값을 둔다.
    }
}

/* 현재 실제 관절 raw로 채널별 최소·최대를 갱신한다. */
void JointSensorCalibrationTest_Update(JointSensorCalibrationTest_t *test,
                                       const uint16_t raw[ROBOT_JOINT_COUNT])
{
    uint32_t joint;  // 측정할 관절 번호를 저장한다.

    if ((test == NULL) || (raw == NULL))
    {
        return;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        if (raw[joint] < test->minimum[joint])
        {
            test->minimum[joint] = raw[joint];  // 새 최소 raw를 저장한다.
        }
        if (raw[joint] > test->maximum[joint])
        {
            test->maximum[joint] = raw[joint];  // 새 최대 raw를 저장한다.
        }
    }
}

/* 기구를 중립에 둔 순간의 실제 raw를 저장한다. */
void JointSensorCalibrationTest_CaptureZero(JointSensorCalibrationTest_t *test,
                                            const uint16_t raw[ROBOT_JOINT_COUNT])
{
    if ((test == NULL) || (raw == NULL))
    {
        return;
    }

    memcpy(test->zero, raw, sizeof(test->zero));  // 18개 중립 raw를 복사한다.
    test->zero_captured = true;                   // 중립 측정을 완료로 표시한다.
}

/* 충분한 실측 범위와 방향으로 적용 가능한 관절 보정표를 만든다. */
bool JointSensorCalibrationTest_Build(const JointSensorCalibrationTest_t *test,
                                      const int8_t direction[ROBOT_JOINT_COUNT],
                                      JointFeedback_Calibration_t table[ROBOT_JOINT_COUNT])
{
    uint32_t joint;  // 보정표를 만들 관절 번호를 저장한다.

    if ((test == NULL) || (direction == NULL) || (table == NULL) ||
        !test->zero_captured)
    {
        return false;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        if ((test->minimum[joint] >= test->zero[joint]) ||
            (test->zero[joint] >= test->maximum[joint]) ||
            ((direction[joint] != 1) && (direction[joint] != -1)))
        {
            return false;
        }

        table[joint].raw_min = test->minimum[joint];                 // 실측 최소 raw를 저장한다.
        table[joint].raw_zero = test->zero[joint];                   // 실측 중립 raw를 저장한다.
        table[joint].raw_max = test->maximum[joint];                 // 실측 최대 raw를 저장한다.
        table[joint].angle_min_rad = ROBOT_JOINT_MIN_RAD;            // 현재 기구 최소각을 적용한다.
        table[joint].angle_zero_rad = 0.0f;                           // 중립 관절각을 0으로 둔다.
        table[joint].angle_max_rad = ROBOT_JOINT_MAX_RAD;            // 현재 기구 최대각을 적용한다.
        table[joint].direction = direction[joint];                   // 관측한 회전 방향을 적용한다.
        table[joint].calibrated = true;                              // 실측 완료를 표시한다.
    }

    return true;
}
