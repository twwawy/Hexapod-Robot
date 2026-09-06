#ifndef MEASUREMENT_STAGE0_H
#define MEASUREMENT_STAGE0_H

#include "stm32f4xx_hal.h"

#include <stdbool.h>

bool MeasurementStage0_Init(UART_HandleTypeDef *gps_uart,
                            UART_HandleTypeDef *imu_uart,
                            SPI_HandleTypeDef *adc_spi,
                            UART_HandleTypeDef *crsf_uart);  // 0단계 센서와 조종기 수신을 준비한다.
void MeasurementStage0_Process(void);                    // 0단계 센서값을 주기적으로 기록한다.
void MeasurementStage0_UartRxCallback(UART_HandleTypeDef *uart);  // 센서·조종기 수신 완료를 전달한다.
void MeasurementStage0_UartErrorCallback(UART_HandleTypeDef *uart);  // 센서·조종기 수신 오류를 전달한다.

#endif
