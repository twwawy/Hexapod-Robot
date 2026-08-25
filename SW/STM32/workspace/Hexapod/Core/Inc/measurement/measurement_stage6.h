#ifndef MEASUREMENT_STAGE6_H
#define MEASUREMENT_STAGE6_H

#include "stm32f4xx_hal.h"

#include <stdbool.h>

typedef enum
{
    MEASUREMENT_STAGE6_WAIT_UNLOADED = 0,  // 무부하 자세 준비 시간을 나타낸다.
    MEASUREMENT_STAGE6_CAPTURE_UNLOADED,   // 무부하 평균 측정을 나타낸다.
    MEASUREMENT_STAGE6_WAIT_LOADED,        // 표시된 발의 누름 준비 시간을 나타낸다.
    MEASUREMENT_STAGE6_CAPTURE_LOADED,     // 표시된 발의 누름 평균 측정을 나타낸다.
    MEASUREMENT_STAGE6_COMPLETE,           // 압력 보정 완료를 나타낸다.
    MEASUREMENT_STAGE6_FAILED              // 압력 보정 실패를 나타낸다.
} MeasurementStage6_Phase_t;

bool MeasurementStage6_Init(SPI_HandleTypeDef *adc_spi);  // 6단계 압력센서 측정을 준비한다.
void MeasurementStage6_Process(void);                     // 무부하와 접촉값을 자동 측정한다.

#endif
