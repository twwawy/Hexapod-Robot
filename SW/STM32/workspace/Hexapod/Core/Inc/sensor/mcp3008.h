#ifndef MCP3008_H
#define MCP3008_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f4xx_hal.h"

#include <stdbool.h>
#include <stdint.h>

#define MCP3008_DEVICE_COUNT     3U
#define MCP3008_CHANNEL_COUNT    8U
#define MCP3008_LEG_COUNT        6U
#define MCP3008_LEG_INPUT_COUNT  4U
#define MCP3008_MAX_RAW_VALUE    1023U
#define MCP3008_SPI_TIMEOUT_MS   10U
#define MCP3008_INVALID_INDEX    0xFFU

typedef enum
{
    MCP3008_DEVICE_1 = 0,
    MCP3008_DEVICE_2,
    MCP3008_DEVICE_3
} MCP3008_Device_t;

/** Column indexes of leg_raw[leg][input]. */
typedef enum
{
    MCP3008_LEG_JOINT_1 = 0,
    MCP3008_LEG_JOINT_2,
    MCP3008_LEG_JOINT_3,
    MCP3008_LEG_PRESSURE
} MCP3008_LegInput_t;

typedef struct
{
    uint8_t device;   // MCP3008 장치 번호를 저장한다.
    uint8_t channel;  // 장치 내부 채널 번호를 저장한다.
} MCP3008_InputMapping_t;

typedef struct
{
    SPI_HandleTypeDef *spi;                                                   // SPI Handle을 저장한다.
    uint32_t timeout_ms;                                                      // SPI 제한 시간을 저장한다.
    MCP3008_InputMapping_t mapping[MCP3008_LEG_COUNT][MCP3008_LEG_INPUT_COUNT]; // 다리 입력별 실제 채널을 저장한다.
} MCP3008_Handle_t;

typedef struct
{
    uint16_t raw[MCP3008_DEVICE_COUNT][MCP3008_CHANNEL_COUNT];

    /**
     * Easy-to-read robot mapping:
     * row    0..5 = leg 1..6
     * column 0..3 = joint 1, joint 2, joint 3, pressure sensor
     */
    uint16_t leg_raw[MCP3008_LEG_COUNT][MCP3008_LEG_INPUT_COUNT];

    uint32_t mcu_time_ms;
    uint32_t update_counter;
    uint32_t error_count;
    uint8_t last_error_device;
    uint8_t last_error_channel;
} MCP3008_Data_t;

/**
 * @brief Bind the driver to SPI1 and set CS1/CS2/CS3 HIGH.
 * @note Call after MX_GPIO_Init() and MX_SPI1_Init().
 */
HAL_StatusTypeDef MCP3008_Init(MCP3008_Handle_t *handle,
                               SPI_HandleTypeDef *spi);

/** Read one single-ended MCP3008 channel. */
HAL_StatusTypeDef MCP3008_ReadChannel(MCP3008_Handle_t *handle,
                                      MCP3008_Device_t device,
                                      uint8_t channel,
                                      uint16_t *raw_value);

/**
 * @brief Read all 24 channels.
 * @note Data is committed only when every channel was read successfully.
 */
HAL_StatusTypeDef MCP3008_ReadAll(MCP3008_Handle_t *handle,
                                  MCP3008_Data_t *data);

bool MCP3008_SetInputMapping(MCP3008_Handle_t *handle,
                             uint8_t leg,
                             MCP3008_LegInput_t input,
                             const MCP3008_InputMapping_t *mapping);  // 다리 입력의 실측 ADC 채널을 설정한다.

/** Safely copy one value from the latest 24-channel snapshot. */
bool MCP3008_GetRaw(const MCP3008_Data_t *data,
                    MCP3008_Device_t device,
                    uint8_t channel,
                    uint16_t *raw_value);

/** Convert a 10-bit raw result to volts using the actual VREF voltage. */
float MCP3008_RawToVoltage(uint16_t raw_value, float vref_v);

#ifdef __cplusplus
}
#endif

#endif /* MCP3008_H */
