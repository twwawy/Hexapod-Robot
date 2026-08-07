#ifndef GPS_H
#define GPS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f4xx_hal.h"

#include <stdbool.h>
#include <stdint.h>

#define GPS_RX_BUFFER_SIZE       512U
#define GPS_NMEA_LINE_SIZE       128U
#define GPS_UBX_MAX_PAYLOAD_SIZE 128U

typedef enum
{
    GPS_PROTOCOL_NONE = 0,
    GPS_PROTOCOL_NMEA,
    GPS_PROTOCOL_UBX
} GPS_Protocol_t;

typedef struct
{
    uint32_t mcu_time_ms;
    uint32_t i_tow_ms;
    uint32_t utc_ms_of_day;

    double latitude_deg;
    double longitude_deg;
    float height_m;

    float velocity_north_mps;
    float velocity_east_mps;
    float velocity_down_mps;
    float ground_speed_mps;
    float heading_motion_rad;

    float horizontal_accuracy_m;
    float vertical_accuracy_m;
    float speed_accuracy_mps;
    float position_dop;

    uint8_t fix_type;
    uint8_t satellites_used;
    bool fix_ok;
    bool position_valid;
    bool velocity_valid;
    bool time_valid;

    GPS_Protocol_t protocol;
    uint32_t update_counter;
} GPS_Data_t;

typedef struct
{
    bool initialized;
    double origin_latitude_deg;
    double origin_longitude_deg;
    double cos_origin_latitude;
} GPS_LocalFrame_t;

typedef struct
{
    UART_HandleTypeDef *uart;
    uint8_t rx_byte;

    uint8_t rx_buffer[GPS_RX_BUFFER_SIZE];
    volatile uint16_t rx_head;
    volatile uint16_t rx_tail;
    volatile uint32_t rx_overflow_count;

    char nmea_line[GPS_NMEA_LINE_SIZE];
    uint16_t nmea_length;
    bool nmea_receiving;

    uint8_t ubx_state;
    uint8_t ubx_class;
    uint8_t ubx_id;
    uint16_t ubx_length;
    uint16_t ubx_index;
    uint8_t ubx_ck_a;
    uint8_t ubx_ck_b;
    uint8_t ubx_payload[GPS_UBX_MAX_PAYLOAD_SIZE];

    GPS_Data_t data;
} GPS_Handle_t;

/**
 * @brief Initialize the GPS parser and bind it to a HAL UART handle.
 * @note  The driver accepts both NMEA GGA/RMC and UBX-NAV-PVT streams.
 */
void GPS_Init(GPS_Handle_t *handle, UART_HandleTypeDef *uart);

/** Start one-byte interrupt reception. The UART IRQ must be enabled. */
HAL_StatusTypeDef GPS_Start(GPS_Handle_t *handle);

/** Call from HAL_UART_RxCpltCallback(). */
void GPS_RxCpltCallback(GPS_Handle_t *handle, UART_HandleTypeDef *uart);

/** Call from HAL_UART_ErrorCallback(). */
void GPS_ErrorCallback(GPS_Handle_t *handle, UART_HandleTypeDef *uart);

/**
 * @brief Parse all bytes queued by the UART interrupt.
 * @return Number of bytes processed.
 * @note  Call frequently from the main loop. Parsing is intentionally kept out
 *        of the UART ISR.
 */
uint32_t GPS_Process(GPS_Handle_t *handle);

/** Feed one byte directly when using DMA or an external transport. */
void GPS_ProcessByte(GPS_Handle_t *handle, uint8_t byte);

/** Copy the most recent complete GPS solution. */
bool GPS_GetLatest(const GPS_Handle_t *handle, GPS_Data_t *out);

/** Quick quality check for navigation use. */
bool GPS_IsUsable(const GPS_Data_t *data, float maximum_horizontal_accuracy_m);

/** Initialize a local North-East frame at the supplied WGS-84 position. */
void GPS_LocalFrame_Init(GPS_LocalFrame_t *frame,
                         double latitude_deg,
                         double longitude_deg);

/** Convert WGS-84 latitude/longitude to local North/East displacement in m. */
bool GPS_LocalFrame_ToNorthEast(const GPS_LocalFrame_t *frame,
                                double latitude_deg,
                                double longitude_deg,
                                float *north_m,
                                float *east_m);

#ifdef __cplusplus
}
#endif

#endif /* GPS_H */
