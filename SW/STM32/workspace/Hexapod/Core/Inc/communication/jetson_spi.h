#ifndef JETSON_SPI_H
#define JETSON_SPI_H

#include "stm32f4xx_hal.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    SPI_HandleTypeDef *spi;        // SPI2 Slave Handle을 저장한다.
    uint32_t transfer_count;       // 완료된 전송 수를 저장한다.
    uint32_t error_count;          // SPI 오류 수를 저장한다.
    bool protocol_ready;           // 상위 프로토콜 확정 여부를 저장한다.
} JetsonSpi_Handle_t;

void JetsonSpi_Init(JetsonSpi_Handle_t *handle,
                    SPI_HandleTypeDef *spi);  // SPI2 최소 인터페이스를 준비한다.

bool JetsonSpi_Process(JetsonSpi_Handle_t *handle);  // 프로토콜 확정 전에는 전송하지 않는다.

#endif
