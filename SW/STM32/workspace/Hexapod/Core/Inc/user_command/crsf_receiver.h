#ifndef CRSF_RECEIVER_H
#define CRSF_RECEIVER_H

#include "stm32f4xx_hal.h"

#include <stdbool.h>
#include <stdint.h>

#define CRSF_RECEIVER_BUFFER_SIZE 256U

typedef struct
{
    UART_HandleTypeDef *uart;                            // USART6 Handle을 저장한다.
    uint8_t rx_byte;                                     // 인터럽트 수신 바이트를 저장한다.
    uint8_t buffer[CRSF_RECEIVER_BUFFER_SIZE];           // 수신 링 버퍼를 저장한다.
    volatile uint16_t head;                              // 쓰기 위치를 저장한다.
    volatile uint16_t tail;                              // 읽기 위치를 저장한다.
    volatile uint32_t overflow_count;                    // 버퍼 초과 횟수를 저장한다.
    volatile uint32_t uart_error_count;                  // UART 오류 횟수를 저장한다.
} CRSF_Receiver_t;

void CRSF_Receiver_Init(CRSF_Receiver_t *receiver,
                        UART_HandleTypeDef *uart);        // USART6 수신 상태를 준비한다.

HAL_StatusTypeDef CRSF_Receiver_Start(CRSF_Receiver_t *receiver);  // 1 byte 인터럽트 수신을 시작한다.

void CRSF_Receiver_RxCpltCallback(CRSF_Receiver_t *receiver,
                                  UART_HandleTypeDef *uart);       // UART 수신 완료를 전달한다.

void CRSF_Receiver_ErrorCallback(CRSF_Receiver_t *receiver,
                                 UART_HandleTypeDef *uart);        // UART 오류를 전달한다.

bool CRSF_Receiver_PopByte(CRSF_Receiver_t *receiver,
                           uint8_t *byte);                          // 저장된 수신 바이트를 꺼낸다.

#endif
