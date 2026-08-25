#ifndef MEASUREMENT_STAGE4_H
#define MEASUREMENT_STAGE4_H

#include "stm32f4xx_hal.h"

#include <stdbool.h>

bool MeasurementStage4_Init(SPI_HandleTypeDef *adc_spi,
                            TIM_HandleTypeDef *tim1,
                            TIM_HandleTypeDef *tim2,
                            TIM_HandleTypeDef *tim3,
                            TIM_HandleTypeDef *tim4,
                            TIM_HandleTypeDef *tim5,
                            TIM_HandleTypeDef *tim8);  // 4단계 전체 서보 보정 영점 시험을 준비한다.
void MeasurementStage4_Process(void);  // 18개 서보를 모두 보정된 0도로 유지한다.

#endif
