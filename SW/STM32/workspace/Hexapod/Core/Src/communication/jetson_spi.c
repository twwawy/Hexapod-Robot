#include "communication/jetson_spi.h"

#include "main.h"

#include <stddef.h>
#include <string.h>

/* SPI2 Slave의 물리 인터페이스만 초기화한다. */
void JetsonSpi_Init(JetsonSpi_Handle_t *handle,
                    SPI_HandleTypeDef *spi)
{
    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));                  // 이전 통신 상태를 제거한다.
    handle->spi = spi;                                   // CubeMX SPI2 Handle을 연결한다.
    handle->protocol_ready = false;                      // 미정 프로토콜을 비활성화한다.
    HAL_GPIO_WritePin(DRDY_GPIO_Port, DRDY_Pin, GPIO_PIN_RESET);  // DRDY를 기본 Low로 둔다.
}

/* 프로토콜 확정 전에는 SPI 송수신을 시작하지 않는다. */
bool JetsonSpi_Process(JetsonSpi_Handle_t *handle)
{
    if ((handle == NULL) || (handle->spi == NULL) || !handle->protocol_ready)
    {
        return false;
    }

    return false;  // 프레임과 DRDY 순서 확정 후 구현한다.
}
