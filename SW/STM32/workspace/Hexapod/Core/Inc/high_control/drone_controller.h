#ifndef DRONE_CONTROLLER_H
#define DRONE_CONTROLLER_H

#include "common/robot_types.h"

typedef enum
{
    DRONE_LANDING_LANDED = 0,  // 완전 착지 상태를 나타낸다.
    DRONE_LANDING_ALL_FEET,    // 전체 발 접촉 대기 상태를 나타낸다.
    DRONE_LANDING_RECOVERY_135,// 1·3·5 복구 상태를 나타낸다.
    DRONE_LANDING_RECOVERY_246,// 2·4·6 복구 상태를 나타낸다.
    DRONE_LANDING_LOWERING,    // 몸체 하강 상태를 나타낸다.
    DRONE_LANDING_SETTLING     // 착지 안정 상태를 나타낸다.
} DroneController_LandingState_t;

typedef struct
{
    RobotControlMode_t previous_mode;           // 이전 제어 모드를 저장한다.
    DroneController_LandingState_t landing_state;  // 착지 세부 상태를 저장한다.
    float posture_memory;                       // 서기 자세 진행률을 저장한다.
    float stand_settle_time_s;                  // 서기 안정 시간을 저장한다.
    float landing_state_time_s;                 // 착지 세부 상태 시간을 저장한다.
    float throttle_filter;                      // Throttle LPF 상태를 저장한다.
    float yaw_filter;                           // Yaw LPF 상태를 저장한다.
    float roll_filter;                          // Roll LPF 상태를 저장한다.
    float pitch_filter;                         // Pitch LPF 상태를 저장한다.
    float yaw_reference_memory;                 // Heading 기준을 저장한다.
    bool stand_complete;                        // 서기 완료 상태를 저장한다.
    bool gait_was_active;                       // 착지 전 보행 여부를 저장한다.
    uint8_t previous_s1;                        // 이전 S1의 세 보행 선택을 저장한다.
} DroneController_Handle_t;

void DroneController_Init(DroneController_Handle_t *handle);  // 모드와 필터 상태를 초기화한다.

RobotDroneOutput_t DroneController_Step(DroneController_Handle_t *handle,
                                        const RobotPriorityOutput_t *priority,
                                        const bool contact[ROBOT_LEG_COUNT],
                                        float yaw_measured_rad);  // 수동·강화학습 등 선택 모드를 제어 명령으로 변환한다.

#endif
