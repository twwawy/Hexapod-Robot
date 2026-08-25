#ifndef MEASUREMENT_STAGE3_H
#define MEASUREMENT_STAGE3_H

#include "stm32f4xx_hal.h"

#include <stdbool.h>

bool MeasurementStage3_Init(SPI_HandleTypeDef *adc_spi);  // 3단계 릴레이 대응 시험을 준비한다.
void MeasurementStage3_Process(void);                     // 릴레이를 순차 출력하고 관절 ADC를 기록한다.

#endif
