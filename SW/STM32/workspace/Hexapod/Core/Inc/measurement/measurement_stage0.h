#ifndef MEASUREMENT_STAGE0_H
#define MEASUREMENT_STAGE0_H

#include "stm32f4xx_hal.h"

#include <stdbool.h>

bool MeasurementStage0_Init(UART_HandleTypeDef *gps_uart,
                            UART_HandleTypeDef *imu_uart,
                            SPI_HandleTypeDef *adc_spi);  // 0단계 실제 센서를 준비한다.
void MeasurementStage0_Process(void);                    // 0단계 센서값을 주기적으로 기록한다.
void MeasurementStage0_UartRxCallback(UART_HandleTypeDef *uart);  // GPS·WT931 수신 완료를 전달한다.
void MeasurementStage0_UartErrorCallback(UART_HandleTypeDef *uart);  // GPS·WT931 수신 오류를 전달한다.

#endif
