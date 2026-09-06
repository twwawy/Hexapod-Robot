#ifndef ROBOT_TYPES_H
#define ROBOT_TYPES_H

#include "common/robot_config.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    float x;   // X축 값을 저장한다.
    float y;   // Y축 값을 저장한다.
    float z;   // Z축 값을 저장한다.
} RobotVec3_t;

typedef enum
{
    ROBOT_CONTROLLER_FAULT_NONE = 0,       // 기록된 제어기 Fault가 없음을 나타낸다.
    ROBOT_CONTROLLER_FAULT_IK_INPUT,       // IK 입력 좌표 오류를 나타낸다.
    ROBOT_CONTROLLER_FAULT_IK_SOLVE        // 최종 IK 해 실패를 나타낸다.
} RobotControllerFaultReason_t;

typedef struct
{
    RobotVec3_t target_body;                 // 최초 실패 발 목표를 저장한다.
    RobotVec3_t limited_body;                // 최초 실패 제한 결과를 저장한다.
    uint32_t control_count;                  // 최초 실패 제어 주기를 저장한다.
    RobotControllerFaultReason_t reason;     // 최초 실패 원인을 저장한다.
    uint8_t leg;                             // 최초 실패 다리 번호를 저장한다.
    bool was_limited;                        // 최초 실패 전 위치 제한 여부를 저장한다.
    bool valid;                              // 최초 실패 기록 존재 여부를 저장한다.
} RobotControllerFaultRecord_t;

typedef struct
{
    float roll;    // Roll 자세를 저장한다.
    float pitch;   // Pitch 자세를 저장한다.
    float yaw;     // Yaw 자세를 저장한다.
} RobotEuler_t;

typedef struct
{
    float vx;   // X축 선속도를 저장한다.
    float vy;   // Y축 선속도를 저장한다.
    float vz;   // Z축 선속도를 저장한다.
    float wz;   // Yaw 각속도를 저장한다.
} RobotBodyTwist_t;

typedef enum
{
    ROBOT_MODE_LANDING = 0,     // 착지 모드를 나타낸다.
    ROBOT_MODE_STAND,           // 서기 모드를 나타낸다.
    ROBOT_MODE_READY,           // 준비 모드를 나타낸다.
    ROBOT_MODE_MANUAL,          // 수동 보행 모드를 나타낸다.
    ROBOT_MODE_CORRECTION,      // 자세 보정 모드를 나타낸다.
    ROBOT_MODE_FAULT,           // 고장 모드를 나타낸다.
    ROBOT_MODE_KILL,            // 긴급 차단 모드를 나타낸다.
    ROBOT_MODE_ARM,             // SC 매니퓰레이터 모드를 나타낸다.
    ROBOT_MODE_AUTONOMOUS       // 추후 자율주행 모드를 예약한다.
} RobotControlMode_t;

typedef enum
{
    ROBOT_LEG_STANCE = 0,          // 지지 상태를 나타낸다.
    ROBOT_LEG_SWING,               // 공중 이동 상태를 나타낸다.
    ROBOT_LEG_LATE_LANDING,        // 지면 탐색 상태를 나타낸다.
    ROBOT_LEG_RECOVERY_SWING,      // 착지 복구 Swing을 나타낸다.
    ROBOT_LEG_TOUCHDOWN_CANDIDATE, // 접촉 확인 대기 상태를 나타낸다.
    ROBOT_LEG_HOLD                 // 보행 일시정지 상태를 나타낸다.
} RobotLegState_t;

typedef enum
{
    ROBOT_TRIPOD_NORMAL = 0,    // 정상 보행을 나타낸다.
    ROBOT_TRIPOD_LAND_ALL,      // 전체 발 착지를 나타낸다.
    ROBOT_TRIPOD_RECOVERY_135,  // 1·3·5번 다리 복구를 나타낸다.
    ROBOT_TRIPOD_RECOVERY_246   // 2·4·6번 다리 복구를 나타낸다.
} RobotTripodMode_t;

typedef enum
{
    ROBOT_WALK_TRIPOD_TURN = 0,     // S1 하단의 3발 회전 보행을 나타낸다.
    ROBOT_WALK_TRIPOD_LATERAL = 1,  // S1 상단의 3발 평행이동을 나타낸다.
    ROBOT_WALK_WAVE_TURN = 2        // S1 중앙의 한 발 회전 보행을 나타낸다.
} RobotWalkMode_t;

typedef enum
{
    ROBOT_GAIT_TRIPOD = 0,  // 두 그룹이 번갈아 움직이는 3발 보행을 나타낸다.
    ROBOT_GAIT_WAVE         // 다섯 발로 지지하는 개별 다리 보행을 나타낸다.
} RobotGaitPattern_t;

typedef struct
{
    RobotVec3_t acceleration_mps2;       // 몸체 가속도를 저장한다.
    RobotVec3_t angular_velocity_radps;  // 몸체 각속도를 저장한다.
    RobotEuler_t attitude_rad;           // 몸체 자세를 저장한다.
    uint32_t timestamp_ms;               // 최근 수신 시각을 저장한다.
    bool valid;                          // 제어용 데이터 유효성을 저장한다.
} RobotImuState_t;

typedef struct
{
    double latitude_deg;        // 위도를 저장한다.
    double longitude_deg;       // 경도를 저장한다.
    float altitude_m;           // 고도를 저장한다.
    uint32_t timestamp_ms;      // 최근 수신 시각을 저장한다.
    bool valid;                 // 위치 유효성을 저장한다.
} RobotGpsState_t;

typedef struct
{
    RobotImuState_t imu;                                      // IMU 상태를 저장한다.
    RobotGpsState_t gps;                                      // GPS 상태를 저장한다.
    float joint_angle_rad[ROBOT_JOINT_COUNT];                 // 측정 관절각을 저장한다.
    uint16_t joint_raw[ROBOT_JOINT_COUNT];                    // 관절센서 raw 값을 저장한다.
    uint16_t pressure_raw[ROBOT_PRESSURE_COUNT];              // 압력센서 raw 값을 저장한다.
    bool foot_contact_raw[ROBOT_LEG_COUNT];                   // Hysteresis 직후 접촉 후보를 저장한다.
    bool foot_contact[ROBOT_LEG_COUNT];                       // 시간 확인을 마친 접촉을 저장한다.
    uint32_t timestamp_ms;                                    // 스냅샷 시각을 저장한다.
} RobotSensorSnapshot_t;

typedef struct
{
    int16_t throttle;        // Throttle 명령을 저장한다.
    int16_t yaw;             // Yaw 명령을 저장한다.
    int16_t roll;            // Roll 명령을 저장한다.
    int16_t pitch;           // Pitch 명령을 저장한다.
    uint8_t sa;              // SA 상태를 저장한다.
    uint8_t sb;              // SB 상태를 저장한다.
    uint8_t sc;              // SC 상태를 저장한다.
    uint8_t sd;              // SD 상태를 저장한다.
    uint8_t se;              // SE 상태를 저장한다.
    uint8_t s1;              // S1 이동 방식을 저장한다.
    uint32_t timestamp_ms;   // 최근 정상 프레임 시각을 저장한다.
    bool connected;          // CRSF 연결 상태를 저장한다.
    bool motion_armed;       // 입력 동작 허가 상태를 저장한다.
} RobotUserCommand_t;

typedef struct
{
    RobotControlMode_t active_mode;  // 선택된 제어 모드를 저장한다.
    int16_t throttle;                // 전달할 Throttle을 저장한다.
    int16_t yaw;                     // 전달할 Yaw를 저장한다.
    int16_t roll;                    // 전달할 Roll을 저장한다.
    int16_t pitch;                   // 전달할 Pitch를 저장한다.
    uint8_t sa;                      // 전달할 SA 상태를 저장한다.
    uint8_t s1;                      // 전달할 S1 이동 방식을 저장한다.
    bool reset_command;              // 기준값 초기화 요청을 저장한다.
} RobotPriorityOutput_t;

typedef struct
{
    bool kill_enable;                         // 릴레이 차단 요청을 저장한다.
    bool reset_command;                       // 기준값 초기화 요청을 저장한다.
    bool stand_enable;                        // 서기 활성화를 저장한다.
    bool landing_enable;                      // 착지 활성화를 저장한다.
    bool tripod_enable;                       // Tripod 활성화를 저장한다.
    RobotTripodMode_t tripod_mode;            // Tripod 동작 모드를 저장한다.
    RobotGaitPattern_t gait_pattern;          // 요청한 정상 보행 패턴을 저장한다.
    bool posture_enable;                      // 자세 제어 활성화를 저장한다.
    bool manual_enable;                       // 수동 모드 활성화를 저장한다.
    bool correction_enable;                   // 보정 모드 활성화를 저장한다.
    bool body_control_enable;                 // 몸체 피드백 활성화를 저장한다.
    float posture_progress;                   // 서기 진행률을 저장한다.
    float recovery_progress;                  // 착지 복구 진행률을 저장한다.
    float vx_user_mps;                        // 사용자 X속도를 저장한다.
    float vy_user_mps;                        // 사용자 Y속도를 저장한다.
    float wz_user_radps;                      // 사용자 Yaw 속도를 저장한다.
    RobotEuler_t posture_reference_rad;       // 자세 목표를 저장한다.
    RobotVec3_t correction_velocity_mps;      // 보정 이동 속도를 저장한다.
    bool stand_done;                          // 서기 완료를 저장한다.
    bool landing_done;                        // 착지 완료를 저장한다.
} RobotDroneOutput_t;

typedef struct
{
    RobotLegState_t state[ROBOT_LEG_COUNT];        // 다리별 상태를 저장한다.
    float progress[ROBOT_LEG_COUNT];               // 다리별 진행률을 저장한다.
    RobotGaitPattern_t gait_pattern;               // 현재 착지까지 유지할 보행 패턴을 저장한다.
    RobotGaitPattern_t next_phase_pattern;         // 다음 경로 검사에 사용할 보행 패턴을 저장한다.
    bool late_landing_exhausted[ROBOT_LEG_COUNT];  // 다리별 Late Landing 한계 도달을 저장한다.
    bool startup_phase;                            // 시작 위상 여부를 저장한다.
    bool waiting_start;                            // 위상 경로 검사 중 발 고정 여부를 저장한다.
    bool next_phase_preview;                       // 다음 위상 검사 시작을 알린다.
    bool next_phase_startup;                       // 검사할 위상의 최초 보행 여부를 저장한다.
    uint8_t next_phase_swing_mask;                 // 검사할 다음 Swing 그룹을 저장한다.
    uint8_t support_recovery_mask;                 // 재접촉을 기다리는 기존 지지발을 저장한다.
    bool late_landing_hold;                        // 탐색 한계 후 위치 고정을 저장한다.
    bool support_recovery_active;                  // 지지발 재착지 일시정지를 저장한다.
    bool enabled_internal;                         // 내부 보행 상태를 저장한다.
    bool late_landing_stop;                        // Late Landing 한계 정지를 저장한다.
} RobotGaitPhase_t;

typedef struct
{
    RobotVec3_t foot[ROBOT_LEG_COUNT];  // 여섯 발 위치를 저장한다.
    bool command_accepted;              // 후보 명령 채택 여부를 저장한다.
} RobotFootTargets_t;

typedef struct
{
    float angle_rad[ROBOT_JOINT_COUNT];  // 18개 관절각을 저장한다.
    bool ik_valid[ROBOT_LEG_COUNT];      // 다리별 IK 유효성을 저장한다.
} RobotJointCommand_t;

typedef struct
{
    bool rollover_fault;    // 전복 Fault를 저장한다.
    bool controller_fault;  // 제어기 Fault를 저장한다.
} RobotSafetyOutput_t;

#endif
