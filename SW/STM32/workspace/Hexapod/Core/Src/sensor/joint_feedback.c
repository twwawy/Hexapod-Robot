#include "sensor/joint_feedback.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

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

/* 세 입력의 가운데 값을 선택한다. */
static float JointFeedback_Median3(float first, float second, float third)
{
    float temporary;  // 값 교환에 사용한다.

    if (first > second)
    {
        temporary = first;  // 첫 값을 보존한다.
        first = second;     // 작은 값을 앞으로 옮긴다.
        second = temporary; // 큰 값을 뒤로 옮긴다.
    }
    if (second > third)
    {
        temporary = second; // 둘째 값을 보존한다.
        second = third;     // 작은 값을 가운데로 옮긴다.
        third = temporary;  // 큰 값을 뒤로 옮긴다.
    }
    if (first > second)
    {
        temporary = first;  // 첫 값을 보존한다.
        first = second;     // 작은 값을 앞으로 옮긴다.
        second = temporary; // 가운데 값을 선택한다.
    }

    return second;
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

    memset(handle, 0, sizeof(*handle));  // 보정표와 필터 상태를 초기화한다.
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

/* 보정 ADC와 PWM 예측을 결합해 최종 관절각을 갱신한다. */
bool JointFeedback_UpdateEstimate(
    JointFeedback_Handle_t *handle,
    const float measured_angle_rad[ROBOT_JOINT_COUNT],
    const float pwm_angle_rad[ROBOT_JOINT_COUNT],
    bool pwm_valid,
    float estimated_angle_rad[ROBOT_JOINT_COUNT])
{
    const float maximum_prediction_step =
        ROBOT_JOINT_PWM_PREDICTION_RATE_RADPS * ROBOT_CONTROL_PERIOD_S;  // 주기당 PWM 예측 한계를 계산한다.
    const uint32_t history_index = handle != NULL ?
        (handle->filter_sample_count % 3U) : 0U;                          // 이번 Median 저장 위치를 계산한다.
    bool valid = true;                                                   // 전체 입력 유효성을 저장한다.
    uint32_t joint;                                                       // 갱신할 관절 번호를 저장한다.

    if ((handle == NULL) || (measured_angle_rad == NULL) ||
        (estimated_angle_rad == NULL) || (pwm_valid && (pwm_angle_rad == NULL)))
    {
        return false;
    }

    if (handle->filter_sample_count == 0U)
    {
        for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
        {
            const float measured = isfinite(measured_angle_rad[joint]) ?
                measured_angle_rad[joint] : 0.0f;  // 첫 비정상 측정은 영점으로 대체한다.
            uint32_t history;                      // 초기화할 Median 위치를 저장한다.

            valid = valid && isfinite(measured_angle_rad[joint]);  // 첫 측정 유효성을 누적한다.
            for (history = 0U; history < 3U; ++history)
            {
                handle->adc_history_rad[joint][history] = measured;  // 첫 측정으로 Median 창을 채운다.
            }
            handle->adc_filtered_rad[joint] = measured;              // 첫 ADC 각도로 저역통과를 시작한다.
            handle->estimated_angle_rad[joint] = measured;           // 첫 출력은 ADC 절대각으로 시작한다.
            handle->pwm_prediction_angle_rad[joint] =
                (pwm_valid && isfinite(pwm_angle_rad[joint])) ?
                pwm_angle_rad[joint] : measured;                      // 현재 PWM 예측 기준을 맞춘다.
            estimated_angle_rad[joint] = measured;                    // 초기 추정각을 반환한다.
        }
        handle->filter_sample_count = 1U;          // 첫 필터 표본을 기록한다.
        handle->pwm_prediction_valid = pwm_valid;  // PWM 기준 준비 여부를 기록한다.
        return valid;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        const bool measured_valid = isfinite(measured_angle_rad[joint]);  // ADC 각도 유효성을 검사한다.
        const float measured = measured_valid ? measured_angle_rad[joint] :
            handle->adc_filtered_rad[joint];                             // 비정상 ADC는 직전 필터값으로 대체한다.
        float median;                                                     // Median 3점 결과를 저장한다.
        float prediction;                                                 // PWM 기반 예측각을 저장한다.

        valid = valid && measured_valid;                                  // ADC 유효성을 누적한다.
        handle->adc_history_rad[joint][history_index] = measured;          // 최신 보정각을 Median 창에 넣는다.
        median = JointFeedback_Median3(handle->adc_history_rad[joint][0],
                                       handle->adc_history_rad[joint][1],
                                       handle->adc_history_rad[joint][2]); // 단발성 ADC 튐을 제거한다.
        handle->adc_filtered_rad[joint] +=
            ROBOT_JOINT_ADC_LPF_ALPHA *
            (median - handle->adc_filtered_rad[joint]);                   // Median 각도의 잔노이즈를 줄인다.
        prediction = handle->estimated_angle_rad[joint];                  // 직전 추정각에서 예측을 시작한다.

        if (pwm_valid && isfinite(pwm_angle_rad[joint]))
        {
            if (handle->pwm_prediction_valid)
            {
                const float prediction_step = JointFeedback_Clamp(
                    pwm_angle_rad[joint] - handle->pwm_prediction_angle_rad[joint],
                    -maximum_prediction_step,
                    maximum_prediction_step);                              // PWM 변화를 서보 속도로 제한한다.

                handle->pwm_prediction_angle_rad[joint] += prediction_step;  // 제한된 PWM 기준각을 갱신한다.
                prediction += prediction_step;                               // 빠른 명령 변화를 예측각에 반영한다.
            }
            else
            {
                handle->pwm_prediction_angle_rad[joint] = pwm_angle_rad[joint];  // 새 PWM 기준을 현재 명령에 맞춘다.
            }

            handle->estimated_angle_rad[joint] = prediction +
                ROBOT_JOINT_ADC_CORRECTION_GAIN *
                (handle->adc_filtered_rad[joint] - prediction);             // ADC 절대각으로 예측 오차를 보정한다.
        }
        else
        {
            handle->estimated_angle_rad[joint] = handle->adc_filtered_rad[joint];  // PWM 없이는 ADC 필터각을 사용한다.
        }

        handle->estimated_angle_rad[joint] = JointFeedback_Clamp(
            handle->estimated_angle_rad[joint],
            ROBOT_JOINT_MIN_RAD,
            ROBOT_JOINT_MAX_RAD);                                           // 최종 추정각을 관절 범위로 제한한다.
        estimated_angle_rad[joint] = handle->estimated_angle_rad[joint];     // 이번 추정각을 반환한다.
    }

    handle->filter_sample_count++;                 // 정상 갱신 시도를 기록한다.
    handle->pwm_prediction_valid = pwm_valid;       // 다음 주기의 PWM 기준 상태를 기록한다.
    return valid;
}
