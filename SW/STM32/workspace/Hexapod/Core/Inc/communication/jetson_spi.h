#ifndef JETSON_SPI_H
#define JETSON_SPI_H

#include "stm32f4xx_hal.h"

#include <stdbool.h>
#include <stdint.h>

#define JETSON_SPI_FRAME_SIZE  32U

typedef struct
{
    SPI_HandleTypeDef *spi;                     // SPI2 Slave Handle을 저장한다.
    uint8_t tx_frame[JETSON_SPI_FRAME_SIZE];    // 다음 전송에서 Jetson으로 보낼 데이터다.
    uint8_t rx_frame[JETSON_SPI_FRAME_SIZE];    // Jetson에서 마지막으로 받은 데이터다.
    uint32_t transfer_count;                    // 완료된 전송 수를 저장한다.
    uint32_t error_count;                       // SPI 오류 수를 저장한다.
    bool protocol_ready;                        // 링크 테스트 사용 가능 여부다.
} JetsonSpi_Handle_t;

void JetsonSpi_Init(JetsonSpi_Handle_t *handle,
                    SPI_HandleTypeDef *spi);  // SPI2와 32바이트 링크 테스트를 준비한다.

/*
 * SPI Slave 수신을 기다리는 블로킹 함수다.
 * 성공하면 받은 32바이트를 다음 트랜잭션에서 그대로 돌려준다.
 */
bool JetsonSpi_Process(JetsonSpi_Handle_t *handle);

#endif
