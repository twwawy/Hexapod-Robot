#ifndef LORA_H
#define LORA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f4xx_hal.h"

#include <stdbool.h>
#include <stdint.h>

#define LORA_RX_BUFFER_SIZE       512U
#define LORA_LINE_SIZE            320U
#define LORA_MAX_PAYLOAD_SIZE     240U
#define LORA_RESPONSE_SIZE        64U
#define LORA_UART_TIMEOUT_MS      100U
#define LORA_BAUD_PROBE_TIMEOUT_MS 250U

typedef struct
{
    uint16_t source_address;
    uint16_t payload_length;
    char payload[LORA_MAX_PAYLOAD_SIZE + 1U];
    int16_t rssi_dbm;
    int16_t snr;
    uint32_t mcu_time_ms;
    uint32_t receive_counter;
} LoRa_Message_t;

typedef struct
{
    UART_HandleTypeDef *uart;
    uint8_t rx_byte;

    uint8_t rx_buffer[LORA_RX_BUFFER_SIZE];
    volatile uint16_t rx_head;
    volatile uint16_t rx_tail;
    volatile uint32_t rx_overflow_count;
    volatile uint32_t uart_error_count;

    char line[LORA_LINE_SIZE];
    uint16_t line_length;

    char last_response[LORA_RESPONSE_SIZE];
    uint32_t response_counter;
    uint32_t parse_error_count;

    /* Startup diagnostics. detected_baud is zero when no AT response was
       received at any supported baud rate. */
    uint32_t detected_baud;
    uint32_t baud_probe_count;
    HAL_StatusTypeDef baud_detect_status;

    LoRa_Message_t latest_message;
} LoRa_Handle_t;

/** Initialize the RYLR998 UART driver. */
void LoRa_Init(LoRa_Handle_t *handle, UART_HandleTypeDef *uart);

/**
 * Probe common RYLR998 UART baud rates by transmitting AT and waiting for +OK.
 * Call this after MX_UARTx_Init() and before LoRa_Start(). The detected rate is
 * applied to the UART and stored in handle->detected_baud.
 */
HAL_StatusTypeDef LoRa_AutoBaud(LoRa_Handle_t *handle);

/** Start one-byte UART interrupt reception. */
HAL_StatusTypeDef LoRa_Start(LoRa_Handle_t *handle);

/** Call from HAL_UART_RxCpltCallback(). */
void LoRa_RxCpltCallback(LoRa_Handle_t *handle,
                         UART_HandleTypeDef *uart);

/** Call from HAL_UART_ErrorCallback(). */
void LoRa_ErrorCallback(LoRa_Handle_t *handle,
                        UART_HandleTypeDef *uart);

/** Parse UART bytes and received +RCV messages. Call in the main loop. */
uint32_t LoRa_Process(LoRa_Handle_t *handle);

/** Copy the latest received LoRa payload. */
bool LoRa_GetLatest(const LoRa_Handle_t *handle, LoRa_Message_t *message);

/** Send an AT command. Do not include CR/LF in command. */
HAL_StatusTypeDef LoRa_SendCommand(LoRa_Handle_t *handle,
                                   const char *command);

/**
 * Configure and save the radio settings.
 * Arduino receiver must use a different address and the same network ID and
 * RF parameter values. BAND must also already be identical on both modules.
 */
HAL_StatusTypeDef LoRa_Configure(LoRa_Handle_t *handle,
                                 uint16_t local_address,
                                 uint8_t network_id);

/**
 * @brief Send an ASCII payload to another RYLR998 address.
 *
 * This is the function to use when sending robot data later.
 *
 * Example:
 *   LoRa_SendText(&g_lora, 2U, "100,200,300");
 *
 * Parameter 2U is the Arduino LoRa address.
 * Parameter "100,200,300" is the data to transmit (maximum 240 bytes).
 */
HAL_StatusTypeDef LoRa_SendText(LoRa_Handle_t *handle,
                                uint16_t destination_address,
                                const char *payload);

#ifdef __cplusplus
}
#endif

#endif /* LORA_H */
