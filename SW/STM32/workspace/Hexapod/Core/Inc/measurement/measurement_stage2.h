#ifndef MEASUREMENT_STAGE2_H
#define MEASUREMENT_STAGE2_H

#include "stm32f4xx_hal.h"

#include <stdbool.h>

bool MeasurementStage2_Init(SPI_HandleTypeDef *adc_spi);  // 2단계 ADC 입력 매핑을 준비한다.
void MeasurementStage2_Process(void);                     // 센서 움직임으로 ADC 채널을 자동 매핑한다.

#endif
