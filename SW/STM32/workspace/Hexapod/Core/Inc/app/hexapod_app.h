#ifndef HEXAPOD_APP_H
#define HEXAPOD_APP_H

#include "app/control_timing_debug.h"
#include "app/pressure_load_calibration.h"
#include "app/robot_bringup.h"
#include "communication/jetson_spi.h"
#include "communication/lora.h"
#include "communication/manipulator_link.h"
#include "communication/robot_telemetry.h"
#include "high_control/body_position_estimator.h"
#include "high_control/body_posture_controller.h"
#include "high_control/control_priority.h"
#include "high_control/drone_controller.h"
#include "high_control/foot_trajectory.h"
#include "high_control/gait_manager.h"
#include "high_control/gait_pose_controller.h"
#include "high_control/leg_kinematics.h"
#include "high_control/rl_controller.h"
#include "high_control/safety.h"
#include "high_control/workspace_limiter.h"
#include "low_control/servo_pwm.h"
#include "sensor/gps.h"
#include "sensor/imu.h"
#include "sensor/mcp3008.h"
#include "sensor/sensor_manager.h"
#include "user_command/crsf_protocol.h"
#include "user_command/crsf_receiver.h"
#include "user_command/user_command.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    UART_HandleTypeDef *gps_uart;          // GPS UART를 연결한다.
    UART_HandleTypeDef *imu_uart;          // WT931 UART를 연결한다.
    UART_HandleTypeDef *lora_uart;         // LoRa UART를 연결한다.
    UART_HandleTypeDef *manipulator_uart;  // 유선 매니퓰레이터 UART를 연결한다.
    UART_HandleTypeDef *crsf_uart;         // CRSF UART를 연결한다.
    SPI_HandleTypeDef *adc_spi;         // MCP3008 SPI를 연결한다.
    SPI_HandleTypeDef *jetson_spi;      // Jetson SPI를 연결한다.
    TIM_HandleTypeDef *control_timer;   // 1 ms TIM6를 연결한다.
    ServoPwm_TimerBank_t servo_timers;  // 서보 PWM 타이머를 연결한다.
    uint16_t lora_local_address;        // 로봇 LoRa 주소를 저장한다.
    uint16_t lora_destination;          // 관제탑 LoRa 주소를 저장한다.
    uint8_t lora_network_id;            // LoRa Network ID를 저장한다.
    bool configure_lora;                // 저장된 LoRa 설정 갱신 여부를 저장한다.
} HexapodApp_Hardware_t;

typedef enum
{
    HEXAPOD_STARTUP_ZERO_WAIT = 0,  // 서기 요청을 기다린다.
    HEXAPOD_STARTUP_ZERO_CAPTURE,   // PWM 없이 초기 관절각을 측정한다.
    HEXAPOD_STARTUP_ZERO_MOVE,      // 측정각에서 0도로 이동한다.
    HEXAPOD_STARTUP_ZERO_READY,     // 0도 완료 후 서기를 허가한다.
    HEXAPOD_STARTUP_ZERO_COMPLETE,  // 기존 제어 동작을 허가한다.
    HEXAPOD_STARTUP_ZERO_FAULT      // PWM 시작 실패를 유지한다.
} HexapodApp_StartupZeroState_t;

typedef enum
{
    HEXAPOD_RL_DISABLED = 0,  // 강화학습 비활성 상태를 나타낸다.
    HEXAPOD_RL_WAITING,       // 정상 정책 또는 다리 계획을 기다린다.
    HEXAPOD_RL_ACTIVE,        // 정상 정책을 제어에 반영한다.
    HEXAPOD_RL_STOPPING,      // 새 이륙 없이 현재 착지를 마친다.
    HEXAPOD_RL_BLOCKED        // SB 재입력이 필요한 오류를 나타낸다.
} HexapodApp_RlState_t;

typedef struct
{
    RobotSensorSnapshot_t sensor;      // 같은 제어 주기의 센서를 저장한다.
    RobotGaitPhase_t gait;             // 같은 제어 주기의 보행 상태를 저장한다.
    FootTrajectory_Plan_t plan;        // 잔차를 적용하기 전의 기본 계획을 저장한다.
    RobotBodyTwist_t applied_twist;    // 현재 위상에 적용 중인 속도를 저장한다.
    float vx_command_mps;             // 전처리한 전후 조종 명령을 저장한다.
    float wz_command_radps;           // 전처리한 회전 조종 명령을 저장한다.
    uint32_t session_id;               // 현재 운용 세션을 저장한다.
    uint32_t timestamp_ms;             // 관측 캡처 시각을 저장한다.
    uint16_t sequence;                 // 관측 순번을 저장한다.
    uint16_t pose_sequence;            // 마지막 자세 목표 소비 순번을 저장한다.
    uint16_t leg_sequence;             // 마지막 위상에 사용한 정책 순번을 저장한다.
    uint16_t leg_plan_id;              // 마지막 실제 적용 계획을 저장한다.
    uint8_t leg_mask;                  // 마지막 잔차 적용 다리를 저장한다.
    bool pose_ack_valid;               // 자세 소비 이력의 유효성을 저장한다.
    bool leg_ack_valid;                // 다리 적용 이력의 유효성을 저장한다.
    HexapodApp_RlState_t state;         // 강화학습 운용 상태를 저장한다.
} HexapodApp_RlObservation_t;

typedef struct
{
    RlController_Handle_t input;              // 통신과 분리한 정책 입력을 검증한다.
    HexapodApp_RlObservation_t observation;   // 외부 입력부에 공개할 최신 관측을 저장한다.
    HexapodApp_RlState_t state;               // 현재 강화학습 운용 상태를 저장한다.
    RobotControlMode_t drain_mode;           // 착지를 마칠 기존 명령원을 저장한다.
    RobotGaitPattern_t drain_pattern;        // 착지를 마칠 기존 보행 패턴을 저장한다.
    uint32_t session_counter;                // 현재 부팅 안의 운용 세션을 구분한다.
    uint32_t candidate_received_ms;          // 고정한 다리 후보의 수락 시각을 저장한다.
    uint32_t candidate_observation_ms;       // 고정한 다리 후보의 관측 시각을 저장한다.
    uint16_t candidate_sequence;             // 검증 중인 정책 출력 순번을 저장한다.
    uint16_t candidate_plan_id;              // 검증 중인 기본 계획 번호를 저장한다.
    uint16_t last_leg_attempt_sequence;       // 마지막 다리 검사 후보의 순번을 저장한다.
    uint16_t observation_sequence;           // 다음 관측 순번을 저장한다.
    bool candidate_valid;                    // 검증 중인 다리 후보의 존재를 저장한다.
    bool has_leg_attempt;                    // 같은 후보의 자동 재검사를 차단한다.
    bool stopping;                           // 기존 궤적의 정지 진행 여부를 저장한다.
    bool rearm_required;                     // 오류 후 SB 재입력 필요 여부를 저장한다.
    bool hold_after_exit;                    // 정상 종료 후 발 위치 유지를 저장한다.
} HexapodApp_RlRuntime_t;

typedef struct
{
    HexapodApp_Hardware_t hardware;                    // CubeMX Handle 연결을 저장한다.
    GPS_Handle_t gps;                                  // GPS 수신 상태를 저장한다.
    IMU_Handle_t imu;                                  // WT931 수신 상태를 저장한다.
    MCP3008_Handle_t adc;                              // MCP3008 상태를 저장한다.
    SensorManager_Handle_t sensors;                    // 센서 통합 상태를 저장한다.
    CRSF_Receiver_t crsf_receiver;                     // CRSF UART 상태를 저장한다.
    CRSF_Protocol_t crsf_protocol;                     // CRSF 해석 상태를 저장한다.
    UserCommand_Handle_t user_command;                 // 조종기 보정과 연결 상태를 저장한다.
    ControlPriority_Handle_t priority_control;         // 운용 모드 상태를 저장한다.
    DroneController_Handle_t drone_control;            // 조종기 명령 변환 상태를 저장한다.
    BodyPositionEstimator_Handle_t position_estimator; // FK 몸체 위치 추정 상태를 저장한다.
    GaitPoseController_Handle_t gait_pose_control;     // 보행 속도 제어 상태를 저장한다.
    WorkspaceLimiter_Handle_t workspace_limiter;       // 동적 작업공간 제한 상태를 저장한다.
    GaitManager_Handle_t gait_manager;                 // Tripod 위상 상태를 저장한다.
    FootTrajectory_Handle_t foot_trajectory;           // 발 궤적 상태를 저장한다.
    BodyPostureController_Handle_t posture_control;    // 몸체 자세 제어 상태를 저장한다.
    LegKinematics_Handle_t kinematics;                 // IK 마지막 정상 해를 저장한다.
    Safety_Handle_t safety_control;                    // 복구 없는 Fault Latch를 저장한다.
    ServoPwm_Handle_t servo_pwm;                       // 관절 PWM 출력 상태를 저장한다.
    LoRa_Handle_t lora;                                // LoRa 송수신 상태를 저장한다.
    ManipulatorLink_Handle_t manipulator;              // 유선 매니퓰레이터 송신 상태를 저장한다.
    RobotTelemetry_Handle_t telemetry;                 // 관제 패킷 주기를 저장한다.
    JetsonSpi_Handle_t jetson;                         // Jetson 32바이트 SPI 프로토콜 상태를 저장한다.
    HexapodApp_RlRuntime_t rl;                         // 패킷과 독립적인 강화학습 제어 상태를 저장한다.
    RobotSensorSnapshot_t sensor_snapshot;             // 최근 실제 센서값을 저장한다.
    RobotUserCommand_t user;                           // 최근 안전한 사용자 명령을 저장한다.
    RobotPriorityOutput_t priority;                    // 최근 운용 모드 출력을 저장한다.
    RobotDroneOutput_t drone;                          // 최근 조종 제어 출력을 저장한다.
    RobotGaitPhase_t gait;                             // 최근 Tripod 상태를 저장한다.
    RobotJointCommand_t joints;                        // 최근 관절 명령을 저장한다.
    RobotSafetyOutput_t safety;                        // 최근 Fault 상태를 저장한다.
    RobotControllerFaultRecord_t controller_fault;    // 최초 제어기 Fault 원인을 저장한다.
    RobotBringupStatus_t bringup;                      // 단계별 출력 허가 상태를 저장한다.
    PressureLoadCalibration_Handle_t pressure_calibration;            // 기립 중 압력 자동 보정 상태를 저장한다.
    IMU_LevelCalibration_t imu_level_calibration;                      // 부팅 시 Roll·Pitch 영점 측정 상태를 저장한다.
    HexapodApp_StartupZeroState_t startup_zero_state;                  // 서기 전 영점 정렬 상태를 저장한다.
    float startup_initial_angle_rad[ROBOT_JOINT_COUNT];                // 영점 정렬 시작각을 저장한다.
    float startup_command_angle_rad[ROBOT_JOINT_COUNT];                // 영점 정렬 명령각을 저장한다.
    float startup_angle_sum_rad[ROBOT_JOINT_COUNT];                    // 초기 관절각 합계를 저장한다.
    float startup_sensor_settle_time_s;                                // 센서 전원 안정 시간을 저장한다.
    uint32_t startup_sensor_sample_count;                              // 초기 관절각 측정 횟수를 저장한다.
    volatile bool pressure_due;                                        // 1 ms 압력 읽기 요청을 저장한다.
    volatile bool control_due;                                         // 5 ms 제어 실행 요청을 저장한다.
    uint8_t control_tick_divider;                                      // 1 ms Tick의 제어 분주를 저장한다.
    uint8_t touchdown_control_mask;                                    // 다음 제어까지 접촉 비트를 유지한다.
    uint32_t control_count;                                            // 완료한 제어 주기 수를 저장한다.
    uint32_t missed_control_count;                                     // 중복 Timer 요청 수를 저장한다.
    bool initialized;                                                  // 초기화 완료 여부를 저장한다.
} HexapodApp_Handle_t;

extern HexapodApp_Handle_t g_hexapod_app;  // Live Expressions용 최종 앱 상태를 공개한다.

HAL_StatusTypeDef HexapodApp_Init(HexapodApp_Handle_t *handle,
                                  const HexapodApp_Hardware_t *hardware);  // 모든 실제 장치와 제어 모듈을 준비한다.

void HexapodApp_Process(HexapodApp_Handle_t *handle);  // 통신 해석과 관제 전송을 처리한다.

bool HexapodApp_GetRlObservation(HexapodApp_Handle_t *handle,
                                  HexapodApp_RlObservation_t *observation);  // 메인 루프에서 최신 관측을 공개하고 이력을 남긴다.

RlController_SubmitResult_t HexapodApp_SubmitRlAction(HexapodApp_Handle_t *handle,
                                                     const RobotRlAction_t *action);  // 메인 루프에서 해석 완료한 정책 출력을 제출한다.

bool HexapodApp_RunControlIfDue(HexapodApp_Handle_t *handle);  // 5분주 요청마다 제어 1회를 실행한다.

void HexapodApp_TimerCallback(HexapodApp_Handle_t *handle,
                              TIM_HandleTypeDef *timer);  // 1 ms Timer 완료를 압력과 제어에 분배한다.

void HexapodApp_UartRxCallback(HexapodApp_Handle_t *handle,
                               UART_HandleTypeDef *uart);  // UART 수신 완료를 해당 드라이버에 전달한다.

void HexapodApp_UartErrorCallback(HexapodApp_Handle_t *handle,
                                  UART_HandleTypeDef *uart);  // UART 오류를 해당 드라이버에 전달한다.

void HexapodApp_UartTxCpltCallback(HexapodApp_Handle_t *handle,
                                   UART_HandleTypeDef *uart);  // UART 송신 완료를 해당 드라이버에 전달한다.

HAL_StatusTypeDef HexapodApp_BoardInit(void);  // 현재 CubeMX Handle로 최종 앱을 시작한다.

void HexapodApp_BoardProcess(void);  // 압력·통신·5 ms 제어 요청을 처리한다.

void HexapodApp_BoardTimerCallback(TIM_HandleTypeDef *timer);  // 최종 앱에 Timer 완료를 전달한다.

void HexapodApp_BoardUartRxCallback(UART_HandleTypeDef *uart);  // 최종 앱에 UART 수신 완료를 전달한다.

void HexapodApp_BoardUartErrorCallback(UART_HandleTypeDef *uart);  // 최종 앱에 UART 오류를 전달한다.

void HexapodApp_BoardUartTxCpltCallback(UART_HandleTypeDef *uart);  // 최종 앱에 UART 송신 완료를 전달한다.

#endif
