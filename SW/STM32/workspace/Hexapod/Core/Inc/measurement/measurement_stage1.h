#ifndef MEASUREMENT_STAGE1_H
#define MEASUREMENT_STAGE1_H

#include "stm32f4xx_hal.h"

#include <stdbool.h>

bool MeasurementStage1_Init(UART_HandleTypeDef *imu_uart);  // 1단계 WT931 각도 실측을 준비한다.
void MeasurementStage1_Process(void);                       // WT931 각도와 보정 요청을 처리한다.
void MeasurementStage1_UartRxCallback(UART_HandleTypeDef *uart);  // WT931 수신 완료를 전달한다.
void MeasurementStage1_UartErrorCallback(UART_HandleTypeDef *uart);  // WT931 수신 오류를 전달한다.

#endif
