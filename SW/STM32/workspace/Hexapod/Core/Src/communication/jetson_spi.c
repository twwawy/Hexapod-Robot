#include "communication/jetson_spi.h"

#include "main.h"

#include <stddef.h>
#include <string.h>

static const uint8_t jetson_spi_start_message[JETSON_SPI_FRAME_SIZE] =
{
    'S', 'T', 'M', '3', '2', 'O', 'K', '!'
};

/* SPI2 Slave와 현재 사용 중인 32바이트 링크 테스트를 초기화한다. */
void JetsonSpi_Init(JetsonSpi_Handle_t *handle,
                    SPI_HandleTypeDef *spi)
{
    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));
    handle->spi = spi;

    if (spi == NULL)
    {
        HAL_GPIO_WritePin(DRDY_GPIO_Port, DRDY_Pin, GPIO_PIN_RESET);
        return;
    }

    memcpy(handle->tx_frame,
           jetson_spi_start_message,
           JETSON_SPI_FRAME_SIZE);
    handle->protocol_ready = true;

    /* Jetson에 첫 번째 32바이트 트랜잭션을 시작해도 된다고 알린다. */
    HAL_GPIO_WritePin(DRDY_GPIO_Port, DRDY_Pin, GPIO_PIN_SET);
}

/*
 * 첫 응답은 "STM32OK!"이고, 이후 응답은 직전 트랜잭션에서 받은
 * 32바이트다. 현재 하드웨어 링크 확인용이며 호출 중에는 블로킹된다.
 */
bool JetsonSpi_Process(JetsonSpi_Handle_t *handle)
{
    HAL_StatusTypeDef status;

    if ((handle == NULL) || (handle->spi == NULL) || !handle->protocol_ready)
    {
        return false;
    }

    status = HAL_SPI_TransmitReceive(handle->spi,
                                     handle->tx_frame,
                                     handle->rx_frame,
                                     JETSON_SPI_FRAME_SIZE,
                                     HAL_MAX_DELAY);

    HAL_GPIO_WritePin(DRDY_GPIO_Port, DRDY_Pin, GPIO_PIN_RESET);

    if (status != HAL_OK)
    {
        handle->error_count++;
        return false;
    }

    memcpy(handle->tx_frame,
           handle->rx_frame,
           JETSON_SPI_FRAME_SIZE);
    handle->transfer_count++;

    /* 다음 32바이트 트랜잭션을 받을 준비가 끝났다. */
    HAL_GPIO_WritePin(DRDY_GPIO_Port, DRDY_Pin, GPIO_PIN_SET);
    return true;
}
