#include "test/leg6_test.h"

#include <stddef.h>
#include <string.h>

static const uint16_t leg6_test_targets_us[] =
{
    LEG6_TEST_MAX_US,      // 양의 작은 범위를 시험한다.
    LEG6_TEST_MIN_US,      // 음의 작은 범위를 시험한다.
    LEG6_TEST_NEUTRAL_US   // 마지막에 중립으로 복귀한다.
};

/* 선택한 6번 다리 관절의 PWM Pulse를 갱신한다. */
static void Leg6Test_WritePulse(Leg6Test_Handle_t *handle,
                                uint8_t joint,
                                uint16_t pulse_us)
{
    handle->pulse_us[joint] = pulse_us;  // 최근 Pulse를 저장한다.
    __HAL_TIM_SET_COMPARE(handle->timer[joint],
                          handle->channel[joint],
                          pulse_us);     // 선택 채널 Compare를 갱신한다.
}

/* 한 PWM 채널을 중립 Pulse에서 시작한다. */
static HAL_StatusTypeDef Leg6Test_StartChannel(TIM_HandleTypeDef *timer,
                                               uint32_t channel)
{
    __HAL_TIM_SET_COMPARE(timer, channel, LEG6_TEST_NEUTRAL_US);  // 시작 전 중립을 기록한다.
    return HAL_TIM_PWM_Start(timer, channel);                     // PWM 출력을 시작한다.
}

/* 6번 다리의 세 PWM을 중립에서 시작한다. */
HAL_StatusTypeDef Leg6Test_Start(Leg6Test_Handle_t *handle,
                                 TIM_HandleTypeDef *tim1,
                                 TIM_HandleTypeDef *tim3,
                                 TIM_HandleTypeDef *tim4)
{
    uint32_t joint;  // 시작할 관절 번호를 저장한다.

    if ((handle == NULL) || (tim1 == NULL) ||
        (tim3 == NULL) || (tim4 == NULL))
    {
        return HAL_ERROR;
    }

    memset(handle, 0, sizeof(*handle));  // 이전 시험 상태를 제거한다.
    handle->timer[0] = tim1;             // 6_1 TIM1을 연결한다.
    handle->timer[1] = tim3;             // 6_2 TIM3을 연결한다.
    handle->timer[2] = tim4;             // 6_3 TIM4를 연결한다.
    handle->channel[0] = TIM_CHANNEL_4;  // 6_1 CH4를 연결한다.
    handle->channel[1] = TIM_CHANNEL_4;  // 6_2 CH4를 연결한다.
    handle->channel[2] = TIM_CHANNEL_4;  // 6_3 CH4를 연결한다.

    for (joint = 0U; joint < LEG6_TEST_JOINT_COUNT; ++joint)
    {
        handle->pulse_us[joint] = LEG6_TEST_NEUTRAL_US;  // 최근 Pulse를 중립으로 둔다.
        if (Leg6Test_StartChannel(handle->timer[joint],
                                  handle->channel[joint]) != HAL_OK)
        {
            handle->state = LEG6_TEST_ERROR;  // PWM 시작 실패를 기록한다.
            Leg6Test_Stop(handle);             // 이미 시작한 채널을 정리한다.
            handle->state = LEG6_TEST_ERROR;  // 정리 후 오류 상태를 복원한다.
            return HAL_ERROR;
        }
    }

    handle->last_step_ms = HAL_GetTick();  // 첫 Ramp 기준 시각을 저장한다.
    handle->state = LEG6_TEST_RUNNING;      // 시험 실행을 표시한다.
    return HAL_OK;
}

/* 한 관절씩 +100·-100·중립 Pulse Ramp를 진행한다. */
void Leg6Test_Process(Leg6Test_Handle_t *handle)
{
    uint32_t now_ms;     // 현재 시각을 저장한다.
    uint16_t target_us;  // 현재 목표 Pulse를 저장한다.
    uint16_t current_us; // 현재 Pulse를 저장한다.

    if ((handle == NULL) || (handle->state != LEG6_TEST_RUNNING))
    {
        return;
    }

    now_ms = HAL_GetTick();  // 비차단 시험 시각을 읽는다.
    if (handle->holding != 0U)
    {
        if ((now_ms - handle->hold_start_ms) < LEG6_TEST_HOLD_MS)
        {
            return;
        }

        handle->holding = 0U;  // 현재 목표 유지 상태를 끝낸다.
        handle->target_index++; // 다음 Pulse 목표로 이동한다.
        if (handle->target_index >=
            (sizeof(leg6_test_targets_us) / sizeof(leg6_test_targets_us[0])))
        {
            handle->target_index = 0U;  // 다음 관절의 첫 목표로 돌아간다.
            handle->active_joint++;     // 다음 관절로 이동한다.
            if (handle->active_joint >= LEG6_TEST_JOINT_COUNT)
            {
                handle->state = LEG6_TEST_COMPLETE;  // 전체 관절 완료를 표시한다.
                return;
            }
        }
    }

    if ((now_ms - handle->last_step_ms) < LEG6_TEST_STEP_INTERVAL_MS)
    {
        return;
    }

    handle->last_step_ms = now_ms;                                      // Ramp 기준 시각을 갱신한다.
    target_us = leg6_test_targets_us[handle->target_index];              // 이번 목표를 읽는다.
    current_us = handle->pulse_us[handle->active_joint];                 // 현재 Pulse를 읽는다.
    if (current_us < target_us)
    {
        const uint16_t difference = (uint16_t)(target_us - current_us);  // 양의 남은 차이를 계산한다.
        current_us = (difference < LEG6_TEST_STEP_US) ?
                     target_us : (uint16_t)(current_us + LEG6_TEST_STEP_US);  // 목표까지 작은 Step으로 올린다.
    }
    else if (current_us > target_us)
    {
        const uint16_t difference = (uint16_t)(current_us - target_us);  // 음의 남은 차이를 계산한다.
        current_us = (difference < LEG6_TEST_STEP_US) ?
                     target_us : (uint16_t)(current_us - LEG6_TEST_STEP_US);  // 목표까지 작은 Step으로 내린다.
    }

    Leg6Test_WritePulse(handle, handle->active_joint, current_us);  // 현재 Ramp Pulse를 출력한다.
    if (current_us == target_us)
    {
        handle->holding = 1U;         // 목표 유지 상태를 시작한다.
        handle->hold_start_ms = now_ms;// 목표 도달 시각을 저장한다.
    }
}

/* 6번 다리를 중립으로 되돌리고 세 PWM을 정지한다. */
void Leg6Test_Stop(Leg6Test_Handle_t *handle)
{
    uint32_t joint;  // 정지할 관절 번호를 저장한다.

    if (handle == NULL)
    {
        return;
    }

    for (joint = 0U; joint < LEG6_TEST_JOINT_COUNT; ++joint)
    {
        if (handle->timer[joint] != NULL)
        {
            Leg6Test_WritePulse(handle, (uint8_t)joint, LEG6_TEST_NEUTRAL_US);  // 정지 전 중립을 출력한다.
            (void)HAL_TIM_PWM_Stop(handle->timer[joint],
                                   handle->channel[joint]);                    // 선택 PWM을 정지한다.
        }
    }

    handle->state = LEG6_TEST_IDLE;  // 안전 정지 상태를 표시한다.
}
