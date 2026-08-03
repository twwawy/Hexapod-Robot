#ifndef IMU_H
#define IMU_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f4xx_hal.h"

#include <stdbool.h>
#include <stdint.h>

#define IMU_RX_BUFFER_SIZE 512U
#define IMU_WIT_FRAME_SIZE 11U

#define IMU_VALID_ACCELEROMETER (1UL << 0)
#define IMU_VALID_GYROSCOPE     (1UL << 1)
#define IMU_VALID_ANGLE         (1UL << 2)
#define IMU_VALID_MAGNETOMETER  (1UL << 3)
#define IMU_VALID_QUATERNION    (1UL << 4)

typedef struct
{
    uint32_t mcu_time_ms;

    float acceleration_mps2[3];
    float angular_velocity_radps[3];
    float euler_angle_rad[3];       /* roll, pitch, yaw */
    int16_t magnetic_raw[3];        /* device counts; calibrate before use */
    float quaternion[4];            /* q0, q1, q2, q3 */
    float temperature_c;

    uint32_t valid_mask;
    uint32_t frame_counter;
    uint32_t checksum_error_count;
} IMU_Data_t;

typedef struct
{
    UART_HandleTypeDef *uart;
    uint8_t rx_byte;

    uint8_t rx_buffer[IMU_RX_BUFFER_SIZE];
    volatile uint16_t rx_head;
    volatile uint16_t rx_tail;
    volatile uint32_t rx_overflow_count;

    uint8_t frame[IMU_WIT_FRAME_SIZE];
    uint8_t frame_index;

    IMU_Data_t data;
} IMU_Handle_t;

/** Initialize the WT931/WIT standard-protocol parser. */
void IMU_Init(IMU_Handle_t *handle, UART_HandleTypeDef *uart);

/** Start one-byte interrupt reception. The UART IRQ must be enabled. */
HAL_StatusTypeDef IMU_Start(IMU_Handle_t *handle);

/** Call from HAL_UART_RxCpltCallback(). */
void IMU_RxCpltCallback(IMU_Handle_t *handle, UART_HandleTypeDef *uart);

/** Call from HAL_UART_ErrorCallback(). */
void IMU_ErrorCallback(IMU_Handle_t *handle, UART_HandleTypeDef *uart);

/** Parse bytes queued by the UART ISR. Call frequently from the main loop. */
uint32_t IMU_Process(IMU_Handle_t *handle);

/** Feed one byte directly when using DMA or an external transport. */
void IMU_ProcessByte(IMU_Handle_t *handle, uint8_t byte);

/** Copy the most recent complete sensor values. */
bool IMU_GetLatest(const IMU_Handle_t *handle, IMU_Data_t *out);

/** True after acceleration, angular velocity, and Euler angle frames arrived. */
bool IMU_HasNavigationData(const IMU_Data_t *data);

#ifdef __cplusplus
}
#endif

#endif /* IMU_H */
