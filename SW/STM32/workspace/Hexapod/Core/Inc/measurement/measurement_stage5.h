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
                            TIM_HandleTypeDef *tim8);  // 5단계 관절 ADC 자동 보정을 준비한다.
void MeasurementStage5_Process(void);  // 한 관절씩 ±20도에서 ADC 보정을 진행한다.

#endif
