#ifndef IMU_CALIBRATION_MEASUREMENT_H
#define IMU_CALIBRATION_MEASUREMENT_H

#include "sensor/imu.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    double euler_sum_rad[3];  // 수평 정지 자세 합계를 저장한다.
    uint32_t sample_count;    // 정상 자세 표본 수를 저장한다.
} ImuCalibrationMeasurement_t;

void ImuCalibrationMeasurement_Init(ImuCalibrationMeasurement_t *measurement);  // WT931 Offset 측정을 초기화한다.
bool ImuCalibrationMeasurement_AddSample(ImuCalibrationMeasurement_t *measurement,
                                         const IMU_Data_t *data);  // 수평 정지 표본을 추가한다.
bool ImuCalibrationMeasurement_Build(const ImuCalibrationMeasurement_t *measurement,
                                     const int8_t euler_angle_sign[3],
                                     IMU_Calibration_t *calibration);  // 자세 부호와 평균 Offset을 만든다.

#endif
