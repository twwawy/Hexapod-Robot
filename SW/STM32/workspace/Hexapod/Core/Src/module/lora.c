#include "module/lora.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t LoRa_NextIndex(uint16_t index)
{
    index++;

    if (index >= LORA_RX_BUFFER_SIZE)
    {
        index = 0U;
    }

    return index;
}

static void LoRa_QueueRxByte(LoRa_Handle_t *handle, uint8_t byte)
{
    uint16_t next_head = LoRa_NextIndex(handle->rx_head);

    if (next_head == handle->rx_tail)
    {
        handle->rx_overflow_count++;
        return;
    }

    handle->rx_buffer[handle->rx_head] = byte;
    handle->rx_head = next_head;
}

static bool LoRa_PopRxByte(LoRa_Handle_t *handle, uint8_t *byte)
{
    if (handle->rx_tail == handle->rx_head)
    {
        return false;
    }

    *byte = handle->rx_buffer[handle->rx_tail];
    handle->rx_tail = LoRa_NextIndex(handle->rx_tail);
    return true;
}

static bool LoRa_ParseReceivedLine(LoRa_Handle_t *handle, const char *line)
{
    const char *cursor;
    char *end_pointer;
    unsigned long source_address;
    unsigned long payload_length;
    long rssi;
    long snr;
    LoRa_Message_t message;

    if (strncmp(line, "+RCV=", 5U) != 0)
    {
        return false;
    }

    cursor = &line[5];
    source_address = strtoul(cursor, &end_pointer, 10);

    if ((end_pointer == cursor) || (*end_pointer != ',') ||
        (source_address > 65535UL))
    {
        return false;
    }

    cursor = end_pointer + 1;
    payload_length = strtoul(cursor, &end_pointer, 10);

    if ((end_pointer == cursor) || (*end_pointer != ',') ||
        (payload_length > LORA_MAX_PAYLOAD_SIZE))
    {
        return false;
    }

    cursor = end_pointer + 1;

    if (strlen(cursor) < (payload_length + 3U))
    {
        return false;
    }

    memset(&message, 0, sizeof(message));
    message.source_address = (uint16_t)source_address;
    message.payload_length = (uint16_t)payload_length;
    memcpy(message.payload, cursor, payload_length);
    message.payload[payload_length] = '\0';

    cursor += payload_length;

    if (*cursor != ',')
    {
        return false;
    }

    cursor++;
    rssi = strtol(cursor, &end_pointer, 10);

    if ((end_pointer == cursor) || (*end_pointer != ','))
    {
        return false;
    }

    cursor = end_pointer + 1;
    snr = strtol(cursor, &end_pointer, 10);

    if (end_pointer == cursor)
    {
        return false;
    }

    message.rssi_dbm = (int16_t)rssi;
    message.snr = (int16_t)snr;
    message.mcu_time_ms = HAL_GetTick();
    message.receive_counter = handle->latest_message.receive_counter + 1U;
    handle->latest_message = message;

    return true;
}

static void LoRa_StoreResponse(LoRa_Handle_t *handle, const char *line)
{
    size_t length = strlen(line);

    if (length >= LORA_RESPONSE_SIZE)
    {
        length = LORA_RESPONSE_SIZE - 1U;
    }

    memcpy(handle->last_response, line, length);
    handle->last_response[length] = '\0';
    handle->response_counter++;
}

void LoRa_Init(LoRa_Handle_t *handle, UART_HandleTypeDef *uart)
{
    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));
    handle->uart = uart;
    handle->baud_detect_status = HAL_ERROR;
}

HAL_StatusTypeDef LoRa_AutoBaud(LoRa_Handle_t *handle)
{
    static const uint32_t candidate_baud_rates[] =
    {
        115200U, 9600U, 57600U, 38400U,
        19200U, 4800U, 2400U, 1200U
    };
    static const uint8_t at_command[] = {'A', 'T', '\r', '\n'};
    uint32_t original_baud;
    size_t baud_index;

    if ((handle == NULL) || (handle->uart == NULL))
    {
        return HAL_ERROR;
    }

    original_baud = handle->uart->Init.BaudRate;
    handle->detected_baud = 0U;
    handle->baud_probe_count = 0U;
    handle->baud_detect_status = HAL_ERROR;

    for (baud_index = 0U;
         baud_index < (sizeof(candidate_baud_rates) /
                       sizeof(candidate_baud_rates[0]));
         baud_index++)
    {
        uint32_t start_ms;

        (void)HAL_UART_AbortReceive(handle->uart);

        if (HAL_UART_DeInit(handle->uart) != HAL_OK)
        {
            continue;
        }

        handle->uart->Init.BaudRate = candidate_baud_rates[baud_index];

        if (HAL_UART_Init(handle->uart) != HAL_OK)
        {
            continue;
        }

        handle->baud_probe_count++;
        handle->rx_head = 0U;
        handle->rx_tail = 0U;
        handle->line_length = 0U;
        handle->response_counter = 0U;
        handle->last_response[0] = '\0';
        __HAL_UART_CLEAR_PEFLAG(handle->uart);

        /* Arm one-byte interrupt reception before transmitting AT. At high
           baud rates the module can start replying before a later blocking
           receive call is reached. */
        if (LoRa_Start(handle) != HAL_OK)
        {
            continue;
        }

        if (HAL_UART_Transmit(handle->uart,
                              at_command,
                              sizeof(at_command),
                              LORA_UART_TIMEOUT_MS) != HAL_OK)
        {
            (void)HAL_UART_AbortReceive(handle->uart);
            continue;
        }

        start_ms = HAL_GetTick();

        while ((HAL_GetTick() - start_ms) < LORA_BAUD_PROBE_TIMEOUT_MS)
        {
            (void)LoRa_Process(handle);

            if (strstr(handle->last_response, "+OK") != NULL)
            {
                (void)HAL_UART_AbortReceive(handle->uart);
                handle->detected_baud = candidate_baud_rates[baud_index];
                handle->baud_detect_status = HAL_OK;
                handle->rx_head = 0U;
                handle->rx_tail = 0U;
                handle->line_length = 0U;
                handle->uart_error_count = 0U;
                return HAL_OK;
            }

            HAL_Delay(5U);
        }

        (void)HAL_UART_AbortReceive(handle->uart);
        HAL_Delay(50U);
    }

    /* Leave the UART in its CubeMX-selected state when detection failed so
       reception can still be armed and inspected in the debugger. */
    (void)HAL_UART_DeInit(handle->uart);
    handle->uart->Init.BaudRate = original_baud;
    (void)HAL_UART_Init(handle->uart);
    handle->rx_head = 0U;
    handle->rx_tail = 0U;
    handle->line_length = 0U;
    handle->response_counter = 0U;
    handle->last_response[0] = '\0';
    handle->uart_error_count = 0U;
    return HAL_ERROR;
}

HAL_StatusTypeDef LoRa_Start(LoRa_Handle_t *handle)
{
    if ((handle == NULL) || (handle->uart == NULL))
    {
        return HAL_ERROR;
    }

    return HAL_UART_Receive_IT(handle->uart, &handle->rx_byte, 1U);
}

void LoRa_RxCpltCallback(LoRa_Handle_t *handle,
                         UART_HandleTypeDef *uart)
{
    if ((handle == NULL) || (handle->uart == NULL) ||
        (uart != handle->uart))
    {
        return;
    }

    LoRa_QueueRxByte(handle, handle->rx_byte);
    (void)HAL_UART_Receive_IT(handle->uart, &handle->rx_byte, 1U);
}

void LoRa_ErrorCallback(LoRa_Handle_t *handle,
                        UART_HandleTypeDef *uart)
{
    if ((handle == NULL) || (handle->uart == NULL) ||
        (uart != handle->uart))
    {
        return;
    }

    handle->uart_error_count++;
    (void)HAL_UART_Receive_IT(handle->uart, &handle->rx_byte, 1U);
}

uint32_t LoRa_Process(LoRa_Handle_t *handle)
{
    uint8_t byte;
    uint32_t processed_count = 0U;

    if (handle == NULL)
    {
        return 0U;
    }

    while (LoRa_PopRxByte(handle, &byte))
    {
        processed_count++;

        if (byte == '\r')
        {
            continue;
        }

        if (byte == '\n')
        {
            if (handle->line_length > 0U)
            {
                handle->line[handle->line_length] = '\0';

                if (!LoRa_ParseReceivedLine(handle, handle->line))
                {
                    LoRa_StoreResponse(handle, handle->line);
                }

                handle->line_length = 0U;
            }

            continue;
        }

        if (handle->line_length < (LORA_LINE_SIZE - 1U))
        {
            handle->line[handle->line_length++] = (char)byte;
        }
        else
        {
            handle->line_length = 0U;
            handle->parse_error_count++;
        }
    }

    return processed_count;
}

bool LoRa_GetLatest(const LoRa_Handle_t *handle, LoRa_Message_t *message)
{
    if ((handle == NULL) || (message == NULL) ||
        (handle->latest_message.receive_counter == 0U))
    {
        return false;
    }

    *message = handle->latest_message;
    return true;
}

HAL_StatusTypeDef LoRa_SendCommand(LoRa_Handle_t *handle,
                                   const char *command)
{
    HAL_StatusTypeDef status;
    static const uint8_t line_end[2] = {'\r', '\n'};
    size_t command_length;

    if ((handle == NULL) || (handle->uart == NULL) || (command == NULL))
    {
        return HAL_ERROR;
    }

    command_length = strlen(command);

    if ((command_length == 0U) || (command_length > 255U))
    {
        return HAL_ERROR;
    }

    status = HAL_UART_Transmit(handle->uart,
                               (const uint8_t *)command,
                               (uint16_t)command_length,
                               LORA_UART_TIMEOUT_MS);

    if (status != HAL_OK)
    {
        return status;
    }

    return HAL_UART_Transmit(handle->uart,
                             line_end,
                             sizeof(line_end),
                             LORA_UART_TIMEOUT_MS);
}

HAL_StatusTypeDef LoRa_Configure(LoRa_Handle_t *handle,
                                 uint16_t local_address,
                                 uint8_t network_id)
{
    char command[48];
    HAL_StatusTypeDef status;

    if ((network_id < 3U) ||
        ((network_id > 15U) && (network_id != 18U)))
    {
        return HAL_ERROR;
    }

    status = LoRa_SendCommand(handle, "AT");
    HAL_Delay(100U);

    if (status != HAL_OK)
    {
        return status;
    }

    (void)snprintf(command, sizeof(command),
                   "AT+ADDRESS=%u", (unsigned int)local_address);
    status = LoRa_SendCommand(handle, command);
    HAL_Delay(100U);

    if (status != HAL_OK)
    {
        return status;
    }

    (void)snprintf(command, sizeof(command),
                   "AT+NETWORKID=%u", (unsigned int)network_id);
    status = LoRa_SendCommand(handle, command);
    HAL_Delay(100U);

    if (status != HAL_OK)
    {
        return status;
    }

    status = LoRa_SendCommand(handle, "AT+PARAMETER=9,7,1,12");
    HAL_Delay(100U);

    return status;
}

HAL_StatusTypeDef LoRa_SendText(LoRa_Handle_t *handle,
                                uint16_t destination_address,
                                const char *payload)
{
    char command_header[40];
    static const uint8_t line_end[2] = {'\r', '\n'};
    size_t payload_length;
    int header_length;
    HAL_StatusTypeDef status;

    if ((handle == NULL) || (handle->uart == NULL) || (payload == NULL))
    {
        return HAL_ERROR;
    }

    payload_length = strlen(payload);

    if ((payload_length == 0U) ||
        (payload_length > LORA_MAX_PAYLOAD_SIZE) ||
        (strchr(payload, '\r') != NULL) || (strchr(payload, '\n') != NULL))
    {
        return HAL_ERROR;
    }

    header_length = snprintf(command_header,
                             sizeof(command_header),
                             "AT+SEND=%u,%u,",
                             (unsigned int)destination_address,
                             (unsigned int)payload_length);

    if ((header_length <= 0) ||
        ((size_t)header_length >= sizeof(command_header)))
    {
        return HAL_ERROR;
    }

    status = HAL_UART_Transmit(handle->uart,
                               (const uint8_t *)command_header,
                               (uint16_t)header_length,
                               LORA_UART_TIMEOUT_MS);

    if (status != HAL_OK)
    {
        return status;
    }

    status = HAL_UART_Transmit(handle->uart,
                               (const uint8_t *)payload,
                               (uint16_t)payload_length,
                               LORA_UART_TIMEOUT_MS);

    if (status != HAL_OK)
    {
        return status;
    }

    return HAL_UART_Transmit(handle->uart,
                             line_end,
                             sizeof(line_end),
                             LORA_UART_TIMEOUT_MS);
}
