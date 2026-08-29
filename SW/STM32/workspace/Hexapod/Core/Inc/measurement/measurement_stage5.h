#ifndef MEASUREMENT_STAGE5_H
#define MEASUREMENT_STAGE5_H

#include "stm32f4xx_hal.h"

#include <stdbool.h>

bool MeasurementStage5_Init(SPI_HandleTypeDef *adc_spi,
                            TIM_HandleTypeDef *tim1,
                            TIM_HandleTypeDef *tim2,
                            TIM_HandleTypeDef *tim3,
                            TIM_HandleTypeDef *tim4,
                            TIM_HandleTypeDef *tim5,
                            TIM_HandleTypeDef *tim8);  // L1 J3·L2 J3 ADC 자동 보정을 준비한다.
void MeasurementStage5_Process(void);  // 대상 두 관절의 ±20도 ADC 보정을 진행한다.

#endif
