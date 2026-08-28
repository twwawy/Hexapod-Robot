#include "sensor/joint_feedback.h"

#include <stddef.h>

/* 입력값을 지정한 범위로 제한한다. */
static float JointFeedback_Clamp(float value, float minimum, float maximum)
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

/* 한 관절의 ADC 값을 관절각으로 변환한다. */
static bool JointFeedback_ConvertOne(const JointFeedback_Calibration_t *calibration,
                                     uint16_t raw,
                                     float *angle_rad)
{
    float angle;   // 보간한 관절각을 저장한다.

    if ((calibration == NULL) || (angle_rad == NULL) ||
        (calibration->raw_min >= calibration->raw_zero) ||
        (calibration->raw_zero >= calibration->raw_max))
    {
        return false;
    }

    if (raw <= calibration->raw_zero)
    {
        const float ratio = ((float)raw - (float)calibration->raw_min) /
                            ((float)calibration->raw_zero - (float)calibration->raw_min);  // 아래 구간 비율을 계산한다.

        angle = calibration->angle_min_rad +
                ratio * (calibration->angle_zero_rad - calibration->angle_min_rad);      // 아래 구간을 선형 보간한다.
    }
    else
    {
        const float ratio = ((float)raw - (float)calibration->raw_zero) /
                            ((float)calibration->raw_max - (float)calibration->raw_zero);  // 위 구간 비율을 계산한다.

        angle = calibration->angle_zero_rad +
                ratio * (calibration->angle_max_rad - calibration->angle_zero_rad);      // 위 구간을 선형 보간한다.
    }

    angle = (calibration->direction < 0) ? -angle : angle;               // 관절 방향을 적용한다.
    *angle_rad = JointFeedback_Clamp(angle,
                                     ROBOT_JOINT_MIN_RAD,
                                     ROBOT_JOINT_MAX_RAD);               // 실측점 밖은 관절 한계까지 선형 확장한다.
    return true;
}

/* 측정 전 사용할 안전한 기본 테이블을 준비한다. */
void JointFeedback_Init(JointFeedback_Handle_t *handle)
{
    uint32_t joint;   // 초기화할 관절 번호를 저장한다.

    if (handle == NULL)
    {
        return;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        handle->table[joint].raw_min = 0U;                         // 임시 ADC 최소값을 넣는다.
        handle->table[joint].raw_zero = 512U;                      // 임시 ADC 영점을 넣는다.
        handle->table[joint].raw_max = 1023U;                      // 임시 ADC 최대값을 넣는다.
        handle->table[joint].angle_min_rad = ROBOT_JOINT_MIN_RAD;  // 임시 최소각을 넣는다.
        handle->table[joint].angle_zero_rad = 0.0f;                // 임시 영점각을 넣는다.
        handle->table[joint].angle_max_rad = ROBOT_JOINT_MAX_RAD;  // 임시 최대각을 넣는다.
        handle->table[joint].direction = 1;                        // 임시 정방향을 넣는다.
        handle->table[joint].calibrated = false;                   // 실측 전 상태로 표시한다.
    }
}

/* 한 관절의 실측 보정값을 설정한다. */
bool JointFeedback_SetCalibration(JointFeedback_Handle_t *handle,
                                  uint8_t joint,
                                  const JointFeedback_Calibration_t *calibration)
{
    if ((handle == NULL) || (calibration == NULL) ||
        (joint >= ROBOT_JOINT_COUNT) ||
        (calibration->raw_min >= calibration->raw_zero) ||
        (calibration->raw_zero >= calibration->raw_max))
    {
        return false;
    }

    handle->table[joint] = *calibration;   // 선택한 관절 테이블을 갱신한다.
    return true;
}

/* 한 관절의 ADC 값을 실측각으로 변환한다. */
bool JointFeedback_ConvertJoint(const JointFeedback_Handle_t *handle,
                                uint8_t joint,
                                uint16_t raw,
                                float *angle_rad)
{
    if ((handle == NULL) || (angle_rad == NULL) || (joint >= ROBOT_JOINT_COUNT))
    {
        return false;
    }

    return JointFeedback_ConvertOne(&handle->table[joint], raw, angle_rad);  // 선택 관절 보정을 적용한다.
}

/* 18개 관절센서 raw 값을 관절각으로 변환한다. */
bool JointFeedback_Convert(const JointFeedback_Handle_t *handle,
                           const uint16_t raw[ROBOT_JOINT_COUNT],
                           float angle_rad[ROBOT_JOINT_COUNT])
{
    uint32_t joint;   // 변환할 관절 번호를 저장한다.
    bool valid = true;   // 전체 변환 결과를 저장한다.

    if ((handle == NULL) || (raw == NULL) || (angle_rad == NULL))
    {
        return false;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        if (!JointFeedback_ConvertJoint(handle, (uint8_t)joint,
                                        raw[joint], &angle_rad[joint]))
        {
            angle_rad[joint] = 0.0f;   // 잘못된 테이블은 영점으로 대체한다.
            valid = false;             // 전체 변환 실패를 기록한다.
        }
    }

    return valid;
}
