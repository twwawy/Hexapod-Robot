#include "user_command/crsf_receiver.h"

#include <stddef.h>
#include <string.h>

_Static_assert((CRSF_RECEIVER_BUFFER_SIZE & (CRSF_RECEIVER_BUFFER_SIZE - 1U)) == 0U,
               "CRSF_RECEIVER_BUFFER_SIZE must be a power of two");

/* USART6 수신 상태를 초기화한다. */
void CRSF_Receiver_Init(CRSF_Receiver_t *receiver,
                        UART_HandleTypeDef *uart)
{
    if (receiver == NULL)
    {
        return;
    }

    memset(receiver, 0, sizeof(*receiver));  // 이전 수신 상태를 제거한다.
    receiver->uart = uart;                   // CRSF UART를 연결한다.
}

/* USART6의 1 byte 인터럽트 수신을 시작한다. */
HAL_StatusTypeDef CRSF_Receiver_Start(CRSF_Receiver_t *receiver)
{
    if ((receiver == NULL) || (receiver->uart == NULL))
    {
        return HAL_ERROR;
    }

    return HAL_UART_Receive_IT(receiver->uart, &receiver->rx_byte, 1U);  // 첫 수신을 요청한다.
}

/* UART 인터럽트 바이트를 링 버퍼에 저장한다. */
void CRSF_Receiver_RxCpltCallback(CRSF_Receiver_t *receiver,
                                  UART_HandleTypeDef *uart)
{
    uint16_t next_head;   // 다음 쓰기 위치를 저장한다.

    if ((receiver == NULL) || (receiver->uart == NULL) ||
        (uart != receiver->uart))
    {
        return;
    }

    next_head = (uint16_t)((receiver->head + 1U) & (CRSF_RECEIVER_BUFFER_SIZE - 1U));  // 다음 위치를 계산한다.

    if (next_head == receiver->tail)
    {
        receiver->overflow_count++;  // 가장 최근 초과 상태를 기록한다.
    }
    else
    {
        receiver->buffer[receiver->head] = receiver->rx_byte;  // 수신 바이트를 저장한다.
        receiver->head = next_head;                             // 쓰기 위치를 갱신한다.
    }

    (void)HAL_UART_Receive_IT(receiver->uart, &receiver->rx_byte, 1U);  // 다음 바이트를 요청한다.
}

/* UART 오류를 기록하고 수신을 다시 시작한다. */
void CRSF_Receiver_ErrorCallback(CRSF_Receiver_t *receiver,
                                 UART_HandleTypeDef *uart)
{
    if ((receiver == NULL) || (receiver->uart == NULL) ||
        (uart != receiver->uart))
    {
        return;
    }

    receiver->uart_error_count++;                                      // UART 오류를 기록한다.
    (void)HAL_UART_AbortReceive(receiver->uart);                        // 이전 수신 상태를 해제한다.
    (void)HAL_UART_Receive_IT(receiver->uart, &receiver->rx_byte, 1U);  // 다음 바이트를 다시 요청한다.
}

/* 수신 링 버퍼에서 한 바이트를 꺼낸다. */
bool CRSF_Receiver_PopByte(CRSF_Receiver_t *receiver,
                           uint8_t *byte)
{
    if ((receiver == NULL) || (byte == NULL) ||
        (receiver->tail == receiver->head))
    {
        return false;
    }

    *byte = receiver->buffer[receiver->tail];                                     // 현재 바이트를 반환한다.
    receiver->tail = (uint16_t)((receiver->tail + 1U) &
                                (CRSF_RECEIVER_BUFFER_SIZE - 1U));                 // 읽기 위치를 갱신한다.
    return true;
}
