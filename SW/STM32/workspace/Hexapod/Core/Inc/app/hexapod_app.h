#ifndef HEXAPOD_APP_H
#define HEXAPOD_APP_H

#include "communication/jetson_spi.h"
#include "communication/lora.h"
#include "communication/robot_telemetry.h"
#include "high_control/body_position_estimator.h"
#include "high_control/body_posture_controller.h"
#include "high_control/control_priority.h"
#include "high_control/drone_controller.h"
#include "high_control/foot_trajectory.h"
#include "high_control/gait_manager.h"
#include "high_control/gait_pose_controller.h"
#include "high_control/leg_kinematics.h"
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
    UART_HandleTypeDef *gps_uart;       // GPS UART를 연결한다.
    UART_HandleTypeDef *imu_uart;       // WT931 UART를 연결한다.
    UART_HandleTypeDef *lora_uart;      // LoRa UART를 연결한다.
    UART_HandleTypeDef *crsf_uart;      // CRSF UART를 연결한다.
    SPI_HandleTypeDef *adc_spi;         // MCP3008 SPI를 연결한다.
    SPI_HandleTypeDef *jetson_spi;      // Jetson SPI를 연결한다.
    TIM_HandleTypeDef *control_timer;   // 5 ms TIM6를 연결한다.
    ServoPwm_TimerBank_t servo_timers;  // 서보 PWM 타이머를 연결한다.
    uint16_t lora_local_address;        // 로봇 LoRa 주소를 저장한다.
    uint16_t lora_destination;          // 관제탑 LoRa 주소를 저장한다.
    uint8_t lora_network_id;            // LoRa Network ID를 저장한다.
    bool configure_lora;                // 저장된 LoRa 설정 갱신 여부를 저장한다.
} HexapodApp_Hardware_t;

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
    RobotTelemetry_Handle_t telemetry;                 // 관제 패킷 주기를 저장한다.
    JetsonSpi_Handle_t jetson;                         // Jetson 32바이트 SPI 프로토콜 상태를 저장한다.
    RobotSensorSnapshot_t sensor_snapshot;             // 최근 실제 센서값을 저장한다.
    RobotUserCommand_t user;                           // 최근 안전한 사용자 명령을 저장한다.
    RobotPriorityOutput_t priority;                    // 최근 운용 모드 출력을 저장한다.
    RobotDroneOutput_t drone;                          // 최근 조종 제어 출력을 저장한다.
    RobotGaitPhase_t gait;                             // 최근 Tripod 상태를 저장한다.
    RobotJointCommand_t joints;                        // 최근 관절 명령을 저장한다.
    RobotSafetyOutput_t safety;                        // 최근 Fault 상태를 저장한다.
    volatile bool control_due;                         // 5 ms 제어 실행 요청을 저장한다.
    uint32_t control_count;                            // 완료한 제어 주기 수를 저장한다.
    uint32_t missed_control_count;                     // 중복 Timer 요청 수를 저장한다.
    bool initialized;                                 // 초기화 완료 여부를 저장한다.
} HexapodApp_Handle_t;

HAL_StatusTypeDef HexapodApp_Init(HexapodApp_Handle_t *handle,
                                  const HexapodApp_Hardware_t *hardware);  // 모든 실제 장치와 제어 모듈을 준비한다.

void HexapodApp_Process(HexapodApp_Handle_t *handle);  // 통신 해석과 관제 전송을 처리한다.

bool HexapodApp_RunControlIfDue(HexapodApp_Handle_t *handle);  // TIM6 요청마다 제어 1회를 실행한다.

void HexapodApp_TimerCallback(HexapodApp_Handle_t *handle,
                              TIM_HandleTypeDef *timer);  // 5 ms Timer 완료를 전달한다.

void HexapodApp_UartRxCallback(HexapodApp_Handle_t *handle,
                               UART_HandleTypeDef *uart);  // UART 수신 완료를 해당 드라이버에 전달한다.

void HexapodApp_UartErrorCallback(HexapodApp_Handle_t *handle,
                                  UART_HandleTypeDef *uart);  // UART 오류를 해당 드라이버에 전달한다.

#endif
