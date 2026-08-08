#include "sensor/mcp3008.h"

#include "main.h"

#include <string.h>

typedef struct
{
    GPIO_TypeDef *port;
    uint16_t pin;
} MCP3008_ChipSelect_t;

static const MCP3008_ChipSelect_t mcp3008_chip_selects[MCP3008_DEVICE_COUNT] =
{
    {CS1_GPIO_Port, CS1_Pin},
    {CS2_GPIO_Port, CS2_Pin},
    {CS3_GPIO_Port, CS3_Pin}
};

static bool MCP3008_IsValidDevice(MCP3008_Device_t device)
{
    return ((unsigned int)device < MCP3008_DEVICE_COUNT);
}

static bool MCP3008_IsValidChannel(uint8_t channel)
{
    return (channel < MCP3008_CHANNEL_COUNT);
}

static void MCP3008_DeselectAll(void)
{
    uint8_t device;

    for (device = 0U; device < MCP3008_DEVICE_COUNT; device++)
    {
        HAL_GPIO_WritePin(mcp3008_chip_selects[device].port,
                          mcp3008_chip_selects[device].pin,
                          GPIO_PIN_SET);
    }
}

static void MCP3008_Select(MCP3008_Device_t device)
{
    HAL_GPIO_WritePin(mcp3008_chip_selects[device].port,
                      mcp3008_chip_selects[device].pin,
                      GPIO_PIN_RESET);
}

static void MCP3008_Deselect(MCP3008_Device_t device)
{
    HAL_GPIO_WritePin(mcp3008_chip_selects[device].port,
                      mcp3008_chip_selects[device].pin,
                      GPIO_PIN_SET);
}

HAL_StatusTypeDef MCP3008_Init(MCP3008_Handle_t *handle,
                               SPI_HandleTypeDef *spi)
{
    if ((handle == NULL) || (spi == NULL))
    {
        return HAL_ERROR;
    }

    handle->spi = spi;
    handle->timeout_ms = MCP3008_SPI_TIMEOUT_MS;
    MCP3008_DeselectAll();

    return HAL_OK;
}

HAL_StatusTypeDef MCP3008_ReadChannel(MCP3008_Handle_t *handle,
                                      MCP3008_Device_t device,
                                      uint8_t channel,
                                      uint16_t *raw_value)
{
    uint8_t tx_data[3];
    uint8_t rx_data[3] = {0U, 0U, 0U};
    HAL_StatusTypeDef status;

    if ((handle == NULL) || (handle->spi == NULL) || (raw_value == NULL) ||
        !MCP3008_IsValidDevice(device) || !MCP3008_IsValidChannel(channel))
    {
        return HAL_ERROR;
    }

    /* Start bit, single-ended mode, and three-bit channel selection. */
    tx_data[0] = 0x01U;
    tx_data[1] = (uint8_t)((0x08U | channel) << 4U);
    tx_data[2] = 0x00U;

    MCP3008_DeselectAll();
    MCP3008_Select(device);

    status = HAL_SPI_TransmitReceive(handle->spi,
                                     tx_data,
                                     rx_data,
                                     3U,
                                     handle->timeout_ms);

    MCP3008_Deselect(device);

    if (status == HAL_OK)
    {
        *raw_value = (uint16_t)((((uint16_t)rx_data[1] & 0x03U) << 8U) |
                                (uint16_t)rx_data[2]);
    }

    return status;
}

HAL_StatusTypeDef MCP3008_ReadAll(MCP3008_Handle_t *handle,
                                  MCP3008_Data_t *data)
{
    uint16_t new_raw[MCP3008_DEVICE_COUNT][MCP3008_CHANNEL_COUNT];
    uint8_t device;
    uint8_t channel;
    uint8_t leg;
    uint8_t input;
    HAL_StatusTypeDef status;

    if ((handle == NULL) || (data == NULL))
    {
        return HAL_ERROR;
    }

    for (device = 0U; device < MCP3008_DEVICE_COUNT; device++)
    {
        for (channel = 0U; channel < MCP3008_CHANNEL_COUNT; channel++)
        {
            status = MCP3008_ReadChannel(handle,
                                         (MCP3008_Device_t)device,
                                         channel,
                                         &new_raw[device][channel]);

            if (status != HAL_OK)
            {
                data->error_count++;
                data->last_error_device = device;
                data->last_error_channel = channel;
                return status;
            }
        }
    }

    memcpy(data->raw, new_raw, sizeof(new_raw));

    /* Convert three MCP3008 x eight channels into six legs x four inputs. */
    for (leg = 0U; leg < MCP3008_LEG_COUNT; leg++)
    {
        device = (uint8_t)(leg / 2U);

        for (input = 0U; input < MCP3008_LEG_INPUT_COUNT; input++)
        {
            channel = (uint8_t)(((leg % 2U) * MCP3008_LEG_INPUT_COUNT) + input);
            data->leg_raw[leg][input] = new_raw[device][channel];
        }
    }

    data->mcu_time_ms = HAL_GetTick();
    data->update_counter++;
    data->last_error_device = MCP3008_INVALID_INDEX;
    data->last_error_channel = MCP3008_INVALID_INDEX;

    return HAL_OK;
}

bool MCP3008_GetRaw(const MCP3008_Data_t *data,
                    MCP3008_Device_t device,
                    uint8_t channel,
                    uint16_t *raw_value)
{
    if ((data == NULL) || (raw_value == NULL) ||
        !MCP3008_IsValidDevice(device) || !MCP3008_IsValidChannel(channel))
    {
        return false;
    }

    *raw_value = data->raw[device][channel];
    return true;
}

float MCP3008_RawToVoltage(uint16_t raw_value, float vref_v)
{
    if (raw_value > MCP3008_MAX_RAW_VALUE)
    {
        raw_value = MCP3008_MAX_RAW_VALUE;
    }

    return ((float)raw_value * vref_v) / (float)MCP3008_MAX_RAW_VALUE;
}
