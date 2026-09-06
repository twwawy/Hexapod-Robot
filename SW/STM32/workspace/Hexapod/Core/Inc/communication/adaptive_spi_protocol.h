#ifndef ADAPTIVE_SPI_PROTOCOL_H
#define ADAPTIVE_SPI_PROTOCOL_H
#include "common/robot_types.h"
#define ADAPTIVE_SPI_SIZE 128U
#define ADAPTIVE_SPI_COMMAND 0x32U
#define ADAPTIVE_SPI_OBSERVATION 0x31U
#define ADAPTIVE_SPI_DETAIL 0x35U
#define ADAPTIVE_SPI_CRC_OFFSET 126U
/* Little endian; SI values quantized explicitly. See docs/ADAPTIVE_SPI_V3.md. */
typedef struct {
    uint32_t session_id;
    uint16_t sequence, plan_id;
    uint8_t swing_mask, gait, next_phase, contacts, raw_contacts, state, flags;
    float elapsed_s, duration_s, height_m;
    RobotEuler_t imu;
    RobotBodyTwist_t command, applied;
    float joints[ROBOT_JOINT_COUNT];
    RobotVec3_t nominal[ROBOT_LEG_COUNT];
    uint16_t ack_sequence, ack_plan;
    uint8_t ack_mask, result, planned_gait;
    RobotVec3_t start[ROBOT_LEG_COUNT];
    RobotVec3_t feet[ROBOT_LEG_COUNT];
    RobotEuler_t posture_command;
    uint8_t leg_state[ROBOT_LEG_COUNT];
    uint32_t timestamp_ms;
} AdaptiveSpi_Observation_t;
uint16_t AdaptiveSpi_Crc(const uint8_t *data, unsigned length);
bool AdaptiveSpi_DecodeExecution(const uint8_t frame[128], RobotAdaptiveExecutionPlan_t *plan);
void AdaptiveSpi_EncodeObservation(uint8_t frame[128], const AdaptiveSpi_Observation_t *o, bool detail);
#endif
