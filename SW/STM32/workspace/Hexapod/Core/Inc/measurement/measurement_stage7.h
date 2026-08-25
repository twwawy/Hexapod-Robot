#ifndef MEASUREMENT_STAGE7_H
#define MEASUREMENT_STAGE7_H

#include "stm32f4xx_hal.h"

#include <stdbool.h>

typedef enum
{
    MEASUREMENT_STAGE7_WAIT_CONNECTION = 0,  // CRSF 정상 프레임 연결 대기를 나타낸다.
    MEASUREMENT_STAGE7_WAIT_CENTER,          // 네 짐벌 중립 준비 시간을 나타낸다.
    MEASUREMENT_STAGE7_CAPTURE_CENTER,       // 네 짐벌 중립 측정을 나타낸다.
    MEASUREMENT_STAGE7_WAIT_TARGET,          // 표시된 입력 위치 준비 시간을 나타낸다.
    MEASUREMENT_STAGE7_CAPTURE_TARGET,       // 표시된 입력 위치 측정을 나타낸다.
    MEASUREMENT_STAGE7_COMPLETE,             // CRSF 보정 완료를 나타낸다.
    MEASUREMENT_STAGE7_FAILED                // CRSF 보정 실패를 나타낸다.
} MeasurementStage7_Phase_t;

bool MeasurementStage7_Init(UART_HandleTypeDef *crsf_uart);  // 7단계 CRSF 실측을 준비한다.
void MeasurementStage7_Process(void);                        // CRSF 채널 보정을 자동 진행한다.
void MeasurementStage7_UartRxCallback(UART_HandleTypeDef *uart);  // CRSF 수신 완료를 전달한다.
void MeasurementStage7_UartErrorCallback(UART_HandleTypeDef *uart); // CRSF 수신 오류를 전달한다.

#endif
