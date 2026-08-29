#ifndef JOINT_FEEDBACK_H
#define JOINT_FEEDBACK_H

#include "common/robot_types.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    uint16_t raw_min;         // 최소 관절각의 ADC 값을 저장한다.
    uint16_t raw_zero;        // 영점 관절각의 ADC 값을 저장한다.
    uint16_t raw_max;         // 최대 관절각의 ADC 값을 저장한다.
    float angle_min_rad;      // 최소 관절각을 저장한다.
    float angle_zero_rad;     // 영점 관절각을 저장한다.
    float angle_max_rad;      // 최대 관절각을 저장한다.
    int8_t direction;         // 관절센서 방향을 저장한다.
    bool calibrated;         // 실측 완료 여부를 저장한다.
} JointFeedback_Calibration_t;

typedef struct
{
    JointFeedback_Calibration_t table[ROBOT_JOINT_COUNT];                  // 관절별 보정 테이블을 저장한다.
    float adc_history_rad[ROBOT_JOINT_COUNT][3];                           // Median 3점 입력을 저장한다.
    float adc_filtered_rad[ROBOT_JOINT_COUNT];                             // 저역통과 ADC 각도를 저장한다.
    float estimated_angle_rad[ROBOT_JOINT_COUNT];                          // 융합 관절각을 저장한다.
    float pwm_prediction_angle_rad[ROBOT_JOINT_COUNT];                     // 속도 제한 PWM 각도를 저장한다.
    uint32_t filter_sample_count;                                          // 정상 필터 표본 수를 저장한다.
    bool pwm_prediction_valid;                                             // PWM 예측 초기화 여부를 저장한다.
} JointFeedback_Handle_t;

void JointFeedback_Init(JointFeedback_Handle_t *handle);   // 기본 보정 테이블을 준비한다.

bool JointFeedback_SetCalibration(JointFeedback_Handle_t *handle,
                                  uint8_t joint,
                                  const JointFeedback_Calibration_t *calibration);  // 한 관절 보정값을 갱신한다.

bool JointFeedback_ConvertJoint(const JointFeedback_Handle_t *handle,
                                uint8_t joint,
                                uint16_t raw,
                                float *angle_rad);  // 한 관절 ADC 값을 실측각으로 변환한다.

bool JointFeedback_Convert(const JointFeedback_Handle_t *handle,
                           const uint16_t raw[ROBOT_JOINT_COUNT],
                           float angle_rad[ROBOT_JOINT_COUNT]);  // 전체 ADC 값을 관절각으로 변환한다.

bool JointFeedback_UpdateEstimate(
    JointFeedback_Handle_t *handle,
    const float measured_angle_rad[ROBOT_JOINT_COUNT],
    const float pwm_angle_rad[ROBOT_JOINT_COUNT],
    bool pwm_valid,
    float estimated_angle_rad[ROBOT_JOINT_COUNT]);  // ADC와 PWM으로 최종 관절각을 추정한다.

#endif
