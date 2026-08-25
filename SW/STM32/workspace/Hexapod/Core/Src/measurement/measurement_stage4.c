#include "measurement/measurement_stage4.h"

#include "common/robot_calibration.h"
#include "low_control/relay.h"
#include "low_control/servo_pwm.h"
#include "measurement/measurement_debug.h"
#include "measurement/measurement_runner.h"
#include "sensor/mcp3008.h"

#include <stddef.h>
#include <string.h>

#define MEASUREMENT_STAGE4_ADC_PERIOD_MS 20U

static MCP3008_Handle_t measurement_adc;        // MCP3008 통신 상태를 저장한다.
static MCP3008_Data_t measurement_adc_data;     // 최근 ADC 24채널 값을 저장한다.
static ServoPwm_Handle_t measurement_servo;      // 18개 PWM 채널 배치를 저장한다.
static MeasurementRunner_t measurement_runner;  // 현재 실측 단계를 저장한다.
static uint32_t measurement_last_adc_ms;         // 최근 ADC 실행 시각을 저장한다.
static uint32_t measurement_start_ms;            // 전체 영점 측정 시작 시각을 저장한다.
static bool measurement_initialized;             // 4단계 실행 가능 여부를 저장한다.

static const float measurement_zero_angle_rad[ROBOT_JOINT_COUNT] = {0.0f};  // 18개 관절의 기구학 영점을 저장한다.

/* 모든 릴레이를 끄고 18개 PWM 출력을 정지한다. */
static void MeasurementStage4_StopOutputs(void)
{
    Relay_AllOff();                           // 서보 전원을 먼저 차단한다.
    ServoPwm_Stop(&measurement_servo);        // 18개 PWM 출력을 정지한다.
    g_measurement_debug.relay_state_mask = 0U;  // 모든 릴레이 OFF를 표시한다.
    g_measurement_debug.servo_test_output_active = false;  // 전체 출력 비활성을 표시한다.
}

/* ADC 24채널을 계속 읽어 Live Expressions에 갱신한다. */
static void MeasurementStage4_UpdateAdc(uint32_t now_ms)
{
    uint32_t device;   // 복사할 MCP3008 번호를 저장한다.
    uint32_t channel;  // 복사할 MCP3008 채널을 저장한다.

    if ((now_ms - measurement_last_adc_ms) < MEASUREMENT_STAGE4_ADC_PERIOD_MS)
    {
        return;
    }
    measurement_last_adc_ms = now_ms;  // 이번 ADC 실행 시각을 저장한다.

    if (MCP3008_ReadAll(&measurement_adc, &measurement_adc_data) != HAL_OK)
    {
        MeasurementStage4_StopOutputs();                                     // ADC 오류 시 전체 서보 전원을 차단한다.
        measurement_initialized = false;                                     // 오류 후 자동 재출력을 막는다.
        g_measurement_debug.adc_driver_error_count = measurement_adc_data.error_count;        // ADC 오류 수를 갱신한다.
        g_measurement_debug.adc_last_error_device = measurement_adc_data.last_error_device;   // 실패 장치를 갱신한다.
        g_measurement_debug.adc_last_error_channel = measurement_adc_data.last_error_channel; // 실패 채널을 갱신한다.
        return;
    }

    for (device = 0U; device < MCP3008_DEVICE_COUNT; ++device)
    {
        for (channel = 0U; channel < MCP3008_CHANNEL_COUNT; ++channel)
        {
            g_measurement_debug.adc_raw[device][channel] =
                measurement_adc_data.raw[device][channel];  // 최근 ADC Raw를 디버거에 복사한다.
        }
    }

    g_measurement_debug.last_sample_ms = measurement_adc_data.mcu_time_ms;       // 최근 ADC 시각을 갱신한다.
    g_measurement_debug.adc_update_count = measurement_adc_data.update_counter;  // 정상 ADC 횟수를 갱신한다.
}

/* 18개 PWM에 보정된 0도 명령을 기록한 뒤 모든 다리 전원을 공급한다. */
bool MeasurementStage4_Init(SPI_HandleTypeDef *adc_spi,
                            TIM_HandleTypeDef *tim1,
                            TIM_HandleTypeDef *tim2,
                            TIM_HandleTypeDef *tim3,
                            TIM_HandleTypeDef *tim4,
                            TIM_HandleTypeDef *tim5,
                            TIM_HandleTypeDef *tim8)
{
    const ServoPwm_TimerBank_t timers = {tim1, tim2, tim3, tim4, tim5, tim8};  // CubeMX 타이머를 묶는다.
    uint32_t joint;  // 적용할 관절 번호를 저장한다.

    MeasurementRunner_Init(&measurement_runner);                   // 전역 디버그값을 초기화한다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 0단계를 건너뛴다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 1단계를 건너뛴다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 2단계를 건너뛴다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 3단계를 건너뛴다.
    g_measurement_debug.calibration = g_robot_calibration;         // 앞 단계의 중앙 설정값을 유지한다.
    memset(&measurement_adc_data, 0, sizeof(measurement_adc_data)); // 이전 ADC 값을 제거한다.
    Relay_Init();                                                  // 모든 다리 전원을 차단한다.
    ServoPwm_Init(&measurement_servo, &timers);                    // 18개 PWM 채널을 준비한다.
    measurement_initialized = false;                              // 초기화 완료 전 실행을 막는다.
    g_measurement_debug.active_servo_joint = UINT8_MAX;           // 전체 서보 활성 표시값을 넣는다.
    g_measurement_debug.active_servo_pulse_us = 0U;               // 관절별 Pulse 사용을 표시한다.
    g_measurement_debug.servo_test_active_leg = 0U;               // 전체 다리 시험임을 표시한다.
    g_measurement_debug.servo_test_active_joint = 0U;             // 전체 관절 시험임을 표시한다.
    g_measurement_debug.servo_test_phase = 0U;                    // 보정 0도 고정 단계를 표시한다.
    g_measurement_debug.servo_test_phase_elapsed_seconds = 0U;    // 전체 영점 측정 시간을 초기화한다.
    g_measurement_debug.servo_test_target_pulse_us = 0U;          // 공통 Pulse를 사용하지 않음을 표시한다.
    g_measurement_debug.servo_test_active_joint_raw = 0U;         // 단일 관절 ADC 표시를 사용하지 않는다.
    g_measurement_debug.servo_test_cycle_count = 0U;              // 순차 반복 횟수를 사용하지 않는다.
    g_measurement_debug.servo_test_output_active = false;         // 출력 비활성 상태로 시작한다.
    g_measurement_debug.relay_state_mask = 0U;                    // 모든 릴레이 OFF를 표시한다.

    if ((adc_spi == NULL) || (tim1 == NULL) || (tim2 == NULL) ||
        (tim3 == NULL) || (tim4 == NULL) || (tim5 == NULL) || (tim8 == NULL))
    {
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // 잘못된 CubeMX Handle을 기록한다.
        return false;
    }
    if (MCP3008_Init(&measurement_adc, adc_spi) != HAL_OK)
    {
        g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_ADC;  // ADC 초기화 실패를 기록한다.
        return false;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        if (!ServoPwm_SetCalibration(&measurement_servo,
                                     (uint8_t)joint,
                                     &g_robot_calibration.servo[joint]))
        {
            MeasurementStage4_StopOutputs();  // 잘못된 서보 보정값이면 전체 출력을 막는다.
            g_measurement_debug.initialization_error =
                MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // 서보 보정값 오류를 기록한다.
            return false;
        }
    }

    if (ServoPwm_Start(&measurement_servo) != HAL_OK)
    {
        MeasurementStage4_StopOutputs();  // PWM 시작 실패 시 모든 출력을 정리한다.
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // PWM 시작 실패를 기록한다.
        return false;
    }
    if (!ServoPwm_WriteAngles(&measurement_servo, measurement_zero_angle_rad))
    {
        MeasurementStage4_StopOutputs();  // 보정 영점 변환 실패 시 전체 출력을 정리한다.
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // 보정 영점 계산 실패를 기록한다.
        return false;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        g_measurement_debug.servo_test_joint_pulse_us[joint] =
            measurement_servo.pulse_us[joint];  // 관절별 보정 영점 Pulse를 표시한다.
    }

    Relay_AllOn();  // 모든 보정 영점 PWM이 준비된 뒤 전체 서보 전원을 공급한다.
    measurement_start_ms = HAL_GetTick();                            // 전체 측정 시작 시각을 저장한다.
    measurement_last_adc_ms = measurement_start_ms;                  // 첫 ADC 실행 기준 시각을 저장한다.
    measurement_initialized = true;                                  // 주기 실행을 허용한다.
    g_measurement_debug.servo_test_output_active = true;              // 전체 PWM과 릴레이 활성을 표시한다.
    g_measurement_debug.relay_state_mask = Relay_GetStateMask();      // 전체 릴레이 상태를 표시한다.
    g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_OK;  // 초기화 성공을 기록한다.
    g_measurement_debug.initialization_complete = true;               // 디버거에 준비 완료를 표시한다.
    return true;
}

/* 18개 서보를 보정된 0도로 계속 유지하며 ADC를 갱신한다. */
void MeasurementStage4_Process(void)
{
    const uint32_t now_ms = HAL_GetTick();  // 현재 실측 시각을 저장한다.

    if (!measurement_initialized)
    {
        return;
    }

    MeasurementStage4_UpdateAdc(now_ms);  // ADC 24채널을 계속 갱신한다.
    g_measurement_debug.servo_test_phase_elapsed_seconds =
        (now_ms - measurement_start_ms) / 1000U;  // 전체 영점 측정 시간을 표시한다.
}
