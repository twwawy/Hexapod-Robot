#include "module/leg6_test.h"

#include <string.h>

static const uint16_t leg6_test_targets_us[] =
{
    LEG6_TEST_MAX_US,
    LEG6_TEST_MIN_US,
    LEG6_TEST_NEUTRAL_US
};

static void Leg6Test_WritePulse(Leg6Test_Handle_t *handle,
                                uint8_t joint,
                                uint16_t pulse_us)
{
    handle->pulse_us[joint] = pulse_us;
    __HAL_TIM_SET_COMPARE(handle->timer[joint],
                          handle->channel[joint],
                          pulse_us);
}

static HAL_StatusTypeDef Leg6Test_StartChannel(TIM_HandleTypeDef *timer,
                                               uint32_t channel)
{
    __HAL_TIM_SET_COMPARE(timer, channel, LEG6_TEST_NEUTRAL_US);
    return HAL_TIM_PWM_Start(timer, channel);
}

HAL_StatusTypeDef Leg6Test_Start(Leg6Test_Handle_t *handle,
                                 TIM_HandleTypeDef *tim1,
                                 TIM_HandleTypeDef *tim3,
                                 TIM_HandleTypeDef *tim4)
{
    uint8_t joint;

    if ((handle == NULL) || (tim1 == NULL) ||
        (tim3 == NULL) || (tim4 == NULL))
    {
        return HAL_ERROR;
    }

    memset(handle, 0, sizeof(*handle));

    handle->timer[0] = tim1;
    handle->timer[1] = tim3;
    handle->timer[2] = tim4;
    handle->channel[0] = TIM_CHANNEL_4;
    handle->channel[1] = TIM_CHANNEL_4;
    handle->channel[2] = TIM_CHANNEL_4;

    for (joint = 0U; joint < LEG6_TEST_JOINT_COUNT; joint++)
    {
        handle->pulse_us[joint] = LEG6_TEST_NEUTRAL_US;

        if (Leg6Test_StartChannel(handle->timer[joint],
                                  handle->channel[joint]) != HAL_OK)
        {
            handle->state = LEG6_TEST_ERROR;
            Leg6Test_Stop(handle);
            handle->state = LEG6_TEST_ERROR;
            return HAL_ERROR;
        }
    }

    handle->active_joint = 0U;
    handle->target_index = 0U;
    handle->last_step_ms = HAL_GetTick();
    handle->state = LEG6_TEST_RUNNING;
    return HAL_OK;
}

void Leg6Test_Process(Leg6Test_Handle_t *handle)
{
    uint32_t now_ms;
    uint16_t target_us;
    uint16_t current_us;

    if ((handle == NULL) || (handle->state != LEG6_TEST_RUNNING))
    {
        return;
    }

    now_ms = HAL_GetTick();

    if (handle->holding != 0U)
    {
        if ((now_ms - handle->hold_start_ms) < LEG6_TEST_HOLD_MS)
        {
            return;
        }

        handle->holding = 0U;
        handle->target_index++;

        if (handle->target_index >=
            (sizeof(leg6_test_targets_us) / sizeof(leg6_test_targets_us[0])))
        {
            handle->target_index = 0U;
            handle->active_joint++;

            if (handle->active_joint >= LEG6_TEST_JOINT_COUNT)
            {
                handle->state = LEG6_TEST_COMPLETE;
                return;
            }
        }
    }

    if ((now_ms - handle->last_step_ms) < LEG6_TEST_STEP_INTERVAL_MS)
    {
        return;
    }

    handle->last_step_ms = now_ms;
    target_us = leg6_test_targets_us[handle->target_index];
    current_us = handle->pulse_us[handle->active_joint];

    if (current_us < target_us)
    {
        uint16_t difference = (uint16_t)(target_us - current_us);
        current_us = (difference < LEG6_TEST_STEP_US)
                   ? target_us
                   : (uint16_t)(current_us + LEG6_TEST_STEP_US);
    }
    else if (current_us > target_us)
    {
        uint16_t difference = (uint16_t)(current_us - target_us);
        current_us = (difference < LEG6_TEST_STEP_US)
                   ? target_us
                   : (uint16_t)(current_us - LEG6_TEST_STEP_US);
    }

    Leg6Test_WritePulse(handle, handle->active_joint, current_us);

    if (current_us == target_us)
    {
        handle->holding = 1U;
        handle->hold_start_ms = now_ms;
    }
}

void Leg6Test_Stop(Leg6Test_Handle_t *handle)
{
    uint8_t joint;

    if (handle == NULL)
    {
        return;
    }

    for (joint = 0U; joint < LEG6_TEST_JOINT_COUNT; joint++)
    {
        if (handle->timer[joint] != NULL)
        {
            Leg6Test_WritePulse(handle, joint, LEG6_TEST_NEUTRAL_US);
            (void)HAL_TIM_PWM_Stop(handle->timer[joint],
                                   handle->channel[joint]);
        }
    }

    handle->state = LEG6_TEST_IDLE;
}
