#include "gps.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GPS_PI                 3.14159265358979323846
#define GPS_DEG_TO_RAD         (GPS_PI / 180.0)
#define GPS_EARTH_RADIUS_M     6378137.0
#define GPS_KNOT_TO_MPS        0.514444444f

enum
{
    GPS_UBX_SYNC_1 = 0,
    GPS_UBX_SYNC_2,
    GPS_UBX_CLASS,
    GPS_UBX_ID,
    GPS_UBX_LENGTH_1,
    GPS_UBX_LENGTH_2,
    GPS_UBX_PAYLOAD,
    GPS_UBX_CHECKSUM_A,
    GPS_UBX_CHECKSUM_B
};

_Static_assert((GPS_RX_BUFFER_SIZE & (GPS_RX_BUFFER_SIZE - 1U)) == 0U,
               "GPS_RX_BUFFER_SIZE must be a power of two");

static void GPS_QueueRxByte(GPS_Handle_t *handle, uint8_t byte);
static void GPS_ParseNmeaByte(GPS_Handle_t *handle, uint8_t byte);
static void GPS_ParseNmeaLine(GPS_Handle_t *handle, char *line);
static void GPS_ParseUbxByte(GPS_Handle_t *handle, uint8_t byte);
static void GPS_ParseNavPvt(GPS_Handle_t *handle);

static uint16_t GPS_ReadU16(const uint8_t *data)
{
    return (uint16_t)data[0]
         | ((uint16_t)data[1] << 8);
}

static uint32_t GPS_ReadU32(const uint8_t *data)
{
    return (uint32_t)data[0]
         | ((uint32_t)data[1] << 8)
         | ((uint32_t)data[2] << 16)
         | ((uint32_t)data[3] << 24);
}

static int32_t GPS_ReadI32(const uint8_t *data)
{
    return (int32_t)GPS_ReadU32(data);
}

static int GPS_HexValue(char value)
{
    if ((value >= '0') && (value <= '9'))
    {
        return value - '0';
    }
    if ((value >= 'A') && (value <= 'F'))
    {
        return value - 'A' + 10;
    }
    if ((value >= 'a') && (value <= 'f'))
    {
        return value - 'a' + 10;
    }
    return -1;
}

static bool GPS_CheckNmeaChecksum(const char *line, const char **checksum_separator)
{
    const char *cursor;
    const char *separator;
    uint8_t checksum = 0U;
    int high;
    int low;

    if ((line == NULL) || (line[0] != '$'))
    {
        return false;
    }

    separator = strchr(line, '*');
    if ((separator == NULL) || (separator[1] == '\0') || (separator[2] == '\0'))
    {
        return false;
    }

    for (cursor = line + 1; cursor < separator; ++cursor)
    {
        checksum ^= (uint8_t)*cursor;
    }

    high = GPS_HexValue(separator[1]);
    low = GPS_HexValue(separator[2]);
    if ((high < 0) || (low < 0))
    {
        return false;
    }

    if (checksum_separator != NULL)
    {
        *checksum_separator = separator;
    }

    return checksum == (uint8_t)((high << 4) | low);
}

static bool GPS_ParseUtc(const char *text, uint32_t *utc_ms_of_day)
{
    double value;
    uint32_t hour;
    uint32_t minute;
    double second;

    if ((text == NULL) || (text[0] == '\0') || (utc_ms_of_day == NULL))
    {
        return false;
    }

    value = strtod(text, NULL);
    hour = (uint32_t)(value / 10000.0);
    value -= (double)hour * 10000.0;
    minute = (uint32_t)(value / 100.0);
    second = value - (double)minute * 100.0;

    if ((hour > 23U) || (minute > 59U) || (second < 0.0) || (second >= 61.0))
    {
        return false;
    }

    *utc_ms_of_day = ((hour * 3600U) + (minute * 60U)) * 1000U
                   + (uint32_t)(second * 1000.0 + 0.5);
    return true;
}

static bool GPS_ParseNmeaCoordinate(const char *text,
                                    const char *hemisphere,
                                    double *coordinate_deg)
{
    double raw;
    double degrees;
    double minutes;

    if ((text == NULL) || (text[0] == '\0')
            || (hemisphere == NULL) || (hemisphere[0] == '\0')
            || (coordinate_deg == NULL))
    {
        return false;
    }

    raw = strtod(text, NULL);
    degrees = floor(raw / 100.0);
    minutes = raw - (degrees * 100.0);
    if ((minutes < 0.0) || (minutes >= 60.0))
    {
        return false;
    }

    *coordinate_deg = degrees + (minutes / 60.0);
    if ((hemisphere[0] == 'S') || (hemisphere[0] == 'W'))
    {
        *coordinate_deg = -*coordinate_deg;
    }
    else if ((hemisphere[0] != 'N') && (hemisphere[0] != 'E'))
    {
        return false;
    }

    return true;
}

static bool GPS_MessageTypeIs(const char *field, const char *suffix)
{
    size_t length;

    if ((field == NULL) || (suffix == NULL))
    {
        return false;
    }

    length = strlen(field);
    return (length >= 3U) && (strcmp(&field[length - 3U], suffix) == 0);
}

void GPS_Init(GPS_Handle_t *handle, UART_HandleTypeDef *uart)
{
    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));
    handle->uart = uart;
    handle->ubx_state = GPS_UBX_SYNC_1;
    handle->data.horizontal_accuracy_m = -1.0f;
    handle->data.vertical_accuracy_m = -1.0f;
    handle->data.speed_accuracy_mps = -1.0f;
    handle->data.position_dop = -1.0f;
}

HAL_StatusTypeDef GPS_Start(GPS_Handle_t *handle)
{
    if ((handle == NULL) || (handle->uart == NULL))
    {
        return HAL_ERROR;
    }

    return HAL_UART_Receive_IT(handle->uart, &handle->rx_byte, 1U);
}

void GPS_RxCpltCallback(GPS_Handle_t *handle, UART_HandleTypeDef *uart)
{
    if ((handle == NULL) || (uart != handle->uart))
    {
        return;
    }

    GPS_QueueRxByte(handle, handle->rx_byte);
    (void)HAL_UART_Receive_IT(handle->uart, &handle->rx_byte, 1U);
}

void GPS_ErrorCallback(GPS_Handle_t *handle, UART_HandleTypeDef *uart)
{
    if ((handle == NULL) || (uart != handle->uart))
    {
        return;
    }

    (void)HAL_UART_AbortReceive(handle->uart);
    (void)HAL_UART_Receive_IT(handle->uart, &handle->rx_byte, 1U);
}

static void GPS_QueueRxByte(GPS_Handle_t *handle, uint8_t byte)
{
    uint16_t next_head;

    next_head = (uint16_t)((handle->rx_head + 1U) & (GPS_RX_BUFFER_SIZE - 1U));
    if (next_head == handle->rx_tail)
    {
        handle->rx_overflow_count++;
        return;
    }

    handle->rx_buffer[handle->rx_head] = byte;
    handle->rx_head = next_head;
}

uint32_t GPS_Process(GPS_Handle_t *handle)
{
    uint32_t processed = 0U;
    uint8_t byte;

    if (handle == NULL)
    {
        return 0U;
    }

    while (handle->rx_tail != handle->rx_head)
    {
        byte = handle->rx_buffer[handle->rx_tail];
        handle->rx_tail = (uint16_t)((handle->rx_tail + 1U)
                                    & (GPS_RX_BUFFER_SIZE - 1U));
        GPS_ProcessByte(handle, byte);
        processed++;
    }

    return processed;
}

void GPS_ProcessByte(GPS_Handle_t *handle, uint8_t byte)
{
    if (handle == NULL)
    {
        return;
    }

    GPS_ParseNmeaByte(handle, byte);
    GPS_ParseUbxByte(handle, byte);
}

static void GPS_ParseNmeaByte(GPS_Handle_t *handle, uint8_t byte)
{
    if (byte == '$')
    {
        handle->nmea_receiving = true;
        handle->nmea_length = 0U;
        handle->nmea_line[handle->nmea_length++] = (char)byte;
        return;
    }

    if (!handle->nmea_receiving)
    {
        return;
    }

    if (byte == '\n')
    {
        handle->nmea_line[handle->nmea_length] = '\0';
        GPS_ParseNmeaLine(handle, handle->nmea_line);
        handle->nmea_receiving = false;
        handle->nmea_length = 0U;
        return;
    }

    if (byte == '\r')
    {
        return;
    }

    if (handle->nmea_length >= (GPS_NMEA_LINE_SIZE - 1U))
    {
        handle->nmea_receiving = false;
        handle->nmea_length = 0U;
        return;
    }

    handle->nmea_line[handle->nmea_length++] = (char)byte;
}

static void GPS_ParseNmeaLine(GPS_Handle_t *handle, char *line)
{
    char *fields[24];
    uint32_t field_count = 0U;
    char *cursor;
    const char *checksum_separator;
    double latitude;
    double longitude;
    bool position_ok;

    if (!GPS_CheckNmeaChecksum(line, &checksum_separator))
    {
        return;
    }

    *((char *)checksum_separator) = '\0';
    fields[field_count++] = line + 1;
    for (cursor = line + 1; (*cursor != '\0') && (field_count < 24U); ++cursor)
    {
        if (*cursor == ',')
        {
            *cursor = '\0';
            fields[field_count++] = cursor + 1;
        }
    }

    if (GPS_MessageTypeIs(fields[0], "GGA") && (field_count >= 10U))
    {
        uint32_t fix_quality = (uint32_t)strtoul(fields[6], NULL, 10);

        position_ok = GPS_ParseNmeaCoordinate(fields[2], fields[3], &latitude)
                   && GPS_ParseNmeaCoordinate(fields[4], fields[5], &longitude);

        handle->data.utc_ms_of_day = 0U;
        handle->data.time_valid = GPS_ParseUtc(fields[1], &handle->data.utc_ms_of_day);
        handle->data.fix_ok = (fix_quality > 0U);
        handle->data.fix_type = handle->data.fix_ok ? 3U : 0U;
        handle->data.satellites_used = (uint8_t)strtoul(fields[7], NULL, 10);
        handle->data.position_dop = (fields[8][0] != '\0')
                                  ? strtof(fields[8], NULL) : -1.0f;
        handle->data.height_m = (fields[9][0] != '\0')
                              ? strtof(fields[9], NULL) : 0.0f;
        handle->data.horizontal_accuracy_m = -1.0f;
        handle->data.vertical_accuracy_m = -1.0f;

        if (position_ok)
        {
            handle->data.latitude_deg = latitude;
            handle->data.longitude_deg = longitude;
        }
        handle->data.position_valid = handle->data.fix_ok && position_ok;
        handle->data.protocol = GPS_PROTOCOL_NMEA;
        handle->data.mcu_time_ms = HAL_GetTick();
        handle->data.update_counter++;
    }
    else if (GPS_MessageTypeIs(fields[0], "RMC") && (field_count >= 9U))
    {
        float course_rad;
        float speed_mps;

        position_ok = GPS_ParseNmeaCoordinate(fields[3], fields[4], &latitude)
                   && GPS_ParseNmeaCoordinate(fields[5], fields[6], &longitude);

        handle->data.utc_ms_of_day = 0U;
        handle->data.time_valid = GPS_ParseUtc(fields[1], &handle->data.utc_ms_of_day);
        handle->data.fix_ok = (fields[2][0] == 'A');
        if (handle->data.fix_ok && (handle->data.fix_type == 0U))
        {
            handle->data.fix_type = 2U;
        }

        if (position_ok)
        {
            handle->data.latitude_deg = latitude;
            handle->data.longitude_deg = longitude;
        }
        handle->data.position_valid = handle->data.fix_ok && position_ok;

        speed_mps = (fields[7][0] != '\0')
                  ? strtof(fields[7], NULL) * GPS_KNOT_TO_MPS : 0.0f;
        course_rad = (fields[8][0] != '\0')
                   ? strtof(fields[8], NULL) * (float)GPS_DEG_TO_RAD : 0.0f;
        handle->data.ground_speed_mps = speed_mps;
        handle->data.heading_motion_rad = course_rad;
        handle->data.velocity_north_mps = speed_mps * cosf(course_rad);
        handle->data.velocity_east_mps = speed_mps * sinf(course_rad);
        handle->data.velocity_down_mps = 0.0f;
        handle->data.velocity_valid = handle->data.fix_ok && (fields[7][0] != '\0');
        handle->data.speed_accuracy_mps = -1.0f;
        handle->data.protocol = GPS_PROTOCOL_NMEA;
        handle->data.mcu_time_ms = HAL_GetTick();
        handle->data.update_counter++;
    }
}

static void GPS_UbxChecksumAdd(GPS_Handle_t *handle, uint8_t byte)
{
    handle->ubx_ck_a = (uint8_t)(handle->ubx_ck_a + byte);
    handle->ubx_ck_b = (uint8_t)(handle->ubx_ck_b + handle->ubx_ck_a);
}

static void GPS_UbxReset(GPS_Handle_t *handle, uint8_t possible_sync_byte)
{
    handle->ubx_state = (possible_sync_byte == 0xB5U)
                      ? GPS_UBX_SYNC_2 : GPS_UBX_SYNC_1;
    handle->ubx_index = 0U;
    handle->ubx_length = 0U;
}

static void GPS_ParseUbxByte(GPS_Handle_t *handle, uint8_t byte)
{
    switch (handle->ubx_state)
    {
        case GPS_UBX_SYNC_1:
            if (byte == 0xB5U)
            {
                handle->ubx_state = GPS_UBX_SYNC_2;
            }
            break;

        case GPS_UBX_SYNC_2:
            if (byte == 0x62U)
            {
                handle->ubx_state = GPS_UBX_CLASS;
                handle->ubx_ck_a = 0U;
                handle->ubx_ck_b = 0U;
            }
            else if (byte != 0xB5U)
            {
                handle->ubx_state = GPS_UBX_SYNC_1;
            }
            break;

        case GPS_UBX_CLASS:
            handle->ubx_class = byte;
            GPS_UbxChecksumAdd(handle, byte);
            handle->ubx_state = GPS_UBX_ID;
            break;

        case GPS_UBX_ID:
            handle->ubx_id = byte;
            GPS_UbxChecksumAdd(handle, byte);
            handle->ubx_state = GPS_UBX_LENGTH_1;
            break;

        case GPS_UBX_LENGTH_1:
            handle->ubx_length = byte;
            GPS_UbxChecksumAdd(handle, byte);
            handle->ubx_state = GPS_UBX_LENGTH_2;
            break;

        case GPS_UBX_LENGTH_2:
            handle->ubx_length |= (uint16_t)byte << 8;
            GPS_UbxChecksumAdd(handle, byte);
            handle->ubx_index = 0U;
            if (handle->ubx_length > GPS_UBX_MAX_PAYLOAD_SIZE)
            {
                GPS_UbxReset(handle, byte);
            }
            else
            {
                handle->ubx_state = (handle->ubx_length == 0U)
                                  ? GPS_UBX_CHECKSUM_A : GPS_UBX_PAYLOAD;
            }
            break;

        case GPS_UBX_PAYLOAD:
            handle->ubx_payload[handle->ubx_index++] = byte;
            GPS_UbxChecksumAdd(handle, byte);
            if (handle->ubx_index >= handle->ubx_length)
            {
                handle->ubx_state = GPS_UBX_CHECKSUM_A;
            }
            break;

        case GPS_UBX_CHECKSUM_A:
            if (byte == handle->ubx_ck_a)
            {
                handle->ubx_state = GPS_UBX_CHECKSUM_B;
            }
            else
            {
                GPS_UbxReset(handle, byte);
            }
            break;

        case GPS_UBX_CHECKSUM_B:
            if (byte == handle->ubx_ck_b)
            {
                if ((handle->ubx_class == 0x01U) && (handle->ubx_id == 0x07U))
                {
                    GPS_ParseNavPvt(handle);
                }
            }
            GPS_UbxReset(handle, byte);
            break;

        default:
            GPS_UbxReset(handle, byte);
            break;
    }
}

static void GPS_ParseNavPvt(GPS_Handle_t *handle)
{
    const uint8_t *payload = handle->ubx_payload;
    uint8_t fix_type;
    bool fix_ok;
    bool invalid_llh = false;

    if (handle->ubx_length < 78U)
    {
        return;
    }

    fix_type = payload[20];
    fix_ok = (payload[21] & 0x01U) != 0U;
    if (handle->ubx_length >= 80U)
    {
        invalid_llh = (GPS_ReadU16(&payload[78]) & 0x0001U) != 0U;
    }

    handle->data.i_tow_ms = GPS_ReadU32(&payload[0]);
    handle->data.time_valid = (payload[11] & 0x03U) == 0x03U;
    handle->data.fix_type = fix_type;
    handle->data.fix_ok = fix_ok;
    handle->data.satellites_used = payload[23];

    handle->data.longitude_deg = (double)GPS_ReadI32(&payload[24]) * 1.0e-7;
    handle->data.latitude_deg = (double)GPS_ReadI32(&payload[28]) * 1.0e-7;
    handle->data.height_m = (float)GPS_ReadI32(&payload[36]) * 0.001f;
    handle->data.horizontal_accuracy_m = (float)GPS_ReadU32(&payload[40]) * 0.001f;
    handle->data.vertical_accuracy_m = (float)GPS_ReadU32(&payload[44]) * 0.001f;

    handle->data.velocity_north_mps = (float)GPS_ReadI32(&payload[48]) * 0.001f;
    handle->data.velocity_east_mps = (float)GPS_ReadI32(&payload[52]) * 0.001f;
    handle->data.velocity_down_mps = (float)GPS_ReadI32(&payload[56]) * 0.001f;
    handle->data.ground_speed_mps = (float)GPS_ReadI32(&payload[60]) * 0.001f;
    handle->data.heading_motion_rad = (float)GPS_ReadI32(&payload[64])
                                           * 1.0e-5f * (float)GPS_DEG_TO_RAD;
    handle->data.speed_accuracy_mps = (float)GPS_ReadU32(&payload[68]) * 0.001f;
    handle->data.position_dop = (float)GPS_ReadU16(&payload[76]) * 0.01f;

    handle->data.position_valid = fix_ok && (fix_type >= 2U) && !invalid_llh;
    handle->data.velocity_valid = fix_ok && (fix_type >= 2U);
    handle->data.protocol = GPS_PROTOCOL_UBX;
    handle->data.mcu_time_ms = HAL_GetTick();
    handle->data.update_counter++;
}

bool GPS_GetLatest(const GPS_Handle_t *handle, GPS_Data_t *out)
{
    if ((handle == NULL) || (out == NULL) || (handle->data.update_counter == 0U))
    {
        return false;
    }

    *out = handle->data;
    return true;
}

bool GPS_IsUsable(const GPS_Data_t *data, float maximum_horizontal_accuracy_m)
{
    if ((data == NULL) || !data->fix_ok || !data->position_valid
            || (data->fix_type < 2U))
    {
        return false;
    }

    if ((maximum_horizontal_accuracy_m > 0.0f)
            && (data->horizontal_accuracy_m >= 0.0f)
            && (data->horizontal_accuracy_m > maximum_horizontal_accuracy_m))
    {
        return false;
    }

    return true;
}

void GPS_LocalFrame_Init(GPS_LocalFrame_t *frame,
                         double latitude_deg,
                         double longitude_deg)
{
    if (frame == NULL)
    {
        return;
    }

    frame->origin_latitude_deg = latitude_deg;
    frame->origin_longitude_deg = longitude_deg;
    frame->cos_origin_latitude = cos(latitude_deg * GPS_DEG_TO_RAD);
    frame->initialized = true;
}

bool GPS_LocalFrame_ToNorthEast(const GPS_LocalFrame_t *frame,
                                double latitude_deg,
                                double longitude_deg,
                                float *north_m,
                                float *east_m)
{
    double delta_latitude_rad;
    double delta_longitude_rad;

    if ((frame == NULL) || !frame->initialized
            || (north_m == NULL) || (east_m == NULL))
    {
        return false;
    }

    delta_latitude_rad = (latitude_deg - frame->origin_latitude_deg) * GPS_DEG_TO_RAD;
    delta_longitude_rad = (longitude_deg - frame->origin_longitude_deg) * GPS_DEG_TO_RAD;

    *north_m = (float)(GPS_EARTH_RADIUS_M * delta_latitude_rad);
    *east_m = (float)(GPS_EARTH_RADIUS_M * frame->cos_origin_latitude
                      * delta_longitude_rad);
    return true;
}
