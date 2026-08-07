#include "sensor/nav_kalman.h"

#include <math.h>
#include <string.h>

#define NAV_NORTH_INDEX          0U
#define NAV_EAST_INDEX           1U
#define NAV_VELOCITY_NORTH_INDEX 2U
#define NAV_VELOCITY_EAST_INDEX  3U
#define NAV_MIN_VARIANCE         1.0e-9f

static float NavKalman_Square(float value)
{
    return value * value;
}

static void NavKalman_ResetCovariance(NavKalman_t *filter)
{
    memset(filter->covariance, 0, sizeof(filter->covariance));
    filter->covariance[NAV_NORTH_INDEX][NAV_NORTH_INDEX]
        = NavKalman_Square(filter->config.initial_position_std_m);
    filter->covariance[NAV_EAST_INDEX][NAV_EAST_INDEX]
        = NavKalman_Square(filter->config.initial_position_std_m);
    filter->covariance[NAV_VELOCITY_NORTH_INDEX][NAV_VELOCITY_NORTH_INDEX]
        = NavKalman_Square(filter->config.initial_velocity_std_mps);
    filter->covariance[NAV_VELOCITY_EAST_INDEX][NAV_VELOCITY_EAST_INDEX]
        = NavKalman_Square(filter->config.initial_velocity_std_mps);
}

static void NavKalman_SymmetrizeCovariance(NavKalman_t *filter)
{
    uint32_t row;
    uint32_t column;
    float average;

    for (row = 0U; row < NAV_KALMAN_STATE_SIZE; ++row)
    {
        if (filter->covariance[row][row] < NAV_MIN_VARIANCE)
        {
            filter->covariance[row][row] = NAV_MIN_VARIANCE;
        }

        for (column = row + 1U; column < NAV_KALMAN_STATE_SIZE; ++column)
        {
            average = 0.5f * (filter->covariance[row][column]
                            + filter->covariance[column][row]);
            filter->covariance[row][column] = average;
            filter->covariance[column][row] = average;
        }
    }
}

static bool NavKalman_ScalarUpdate(NavKalman_t *filter,
                                   uint32_t state_index,
                                   float measurement,
                                   float measurement_variance)
{
    float old_covariance[NAV_KALMAN_STATE_SIZE][NAV_KALMAN_STATE_SIZE];
    float gain[NAV_KALMAN_STATE_SIZE];
    float innovation;
    float innovation_variance;
    uint32_t row;
    uint32_t column;

    if (!isfinite(measurement) || !isfinite(measurement_variance)
            || (measurement_variance <= 0.0f))
    {
        return false;
    }

    memcpy(old_covariance, filter->covariance, sizeof(old_covariance));
    innovation = measurement - filter->state[state_index];
    innovation_variance = old_covariance[state_index][state_index]
                        + measurement_variance;
    if (innovation_variance <= NAV_MIN_VARIANCE)
    {
        return false;
    }

    for (row = 0U; row < NAV_KALMAN_STATE_SIZE; ++row)
    {
        gain[row] = old_covariance[row][state_index] / innovation_variance;
        filter->state[row] += gain[row] * innovation;
    }

    for (row = 0U; row < NAV_KALMAN_STATE_SIZE; ++row)
    {
        for (column = 0U; column < NAV_KALMAN_STATE_SIZE; ++column)
        {
            filter->covariance[row][column]
                = old_covariance[row][column]
                - gain[row] * old_covariance[state_index][column];
        }
    }

    NavKalman_SymmetrizeCovariance(filter);
    return true;
}

void NavKalman_DefaultConfig(NavKalman_Config_t *config)
{
    if (config == NULL)
    {
        return;
    }

    config->acceleration_noise_std_mps2 = 1.5f;
    config->initial_position_std_m = 5.0f;
    config->initial_velocity_std_mps = 2.0f;
    config->default_gps_position_std_m = 2.5f;
    config->default_gps_velocity_std_mps = 0.5f;
    config->position_gate_threshold = 9.21f;
}

void NavKalman_Init(NavKalman_t *filter, const NavKalman_Config_t *config)
{
    NavKalman_Config_t default_config;

    if (filter == NULL)
    {
        return;
    }

    memset(filter, 0, sizeof(*filter));
    if (config == NULL)
    {
        NavKalman_DefaultConfig(&default_config);
        filter->config = default_config;
    }
    else
    {
        filter->config = *config;
    }

    NavKalman_ResetCovariance(filter);
}

void NavKalman_SetState(NavKalman_t *filter,
                        float north_m,
                        float east_m,
                        float velocity_north_mps,
                        float velocity_east_mps)
{
    if ((filter == NULL) || !isfinite(north_m) || !isfinite(east_m)
            || !isfinite(velocity_north_mps) || !isfinite(velocity_east_mps))
    {
        return;
    }

    filter->state[NAV_NORTH_INDEX] = north_m;
    filter->state[NAV_EAST_INDEX] = east_m;
    filter->state[NAV_VELOCITY_NORTH_INDEX] = velocity_north_mps;
    filter->state[NAV_VELOCITY_EAST_INDEX] = velocity_east_mps;
    NavKalman_ResetCovariance(filter);
    filter->initialized = true;
}

bool NavKalman_Predict(NavKalman_t *filter,
                       float acceleration_north_mps2,
                       float acceleration_east_mps2,
                       float dt_s)
{
    float transition[NAV_KALMAN_STATE_SIZE][NAV_KALMAN_STATE_SIZE] =
    {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f}
    };
    float temporary[NAV_KALMAN_STATE_SIZE][NAV_KALMAN_STATE_SIZE] = {{0.0f}};
    float predicted_covariance[NAV_KALMAN_STATE_SIZE][NAV_KALMAN_STATE_SIZE] = {{0.0f}};
    float dt2;
    float dt3;
    float dt4;
    float acceleration_variance;
    uint32_t row;
    uint32_t column;
    uint32_t index;

    if ((filter == NULL) || !filter->initialized
            || !isfinite(acceleration_north_mps2)
            || !isfinite(acceleration_east_mps2)
            || !isfinite(dt_s) || (dt_s <= 0.0f) || (dt_s > 1.0f))
    {
        return false;
    }

    dt2 = dt_s * dt_s;
    dt3 = dt2 * dt_s;
    dt4 = dt2 * dt2;

    filter->state[NAV_NORTH_INDEX]
        += filter->state[NAV_VELOCITY_NORTH_INDEX] * dt_s
         + 0.5f * acceleration_north_mps2 * dt2;
    filter->state[NAV_EAST_INDEX]
        += filter->state[NAV_VELOCITY_EAST_INDEX] * dt_s
         + 0.5f * acceleration_east_mps2 * dt2;
    filter->state[NAV_VELOCITY_NORTH_INDEX] += acceleration_north_mps2 * dt_s;
    filter->state[NAV_VELOCITY_EAST_INDEX] += acceleration_east_mps2 * dt_s;

    transition[NAV_NORTH_INDEX][NAV_VELOCITY_NORTH_INDEX] = dt_s;
    transition[NAV_EAST_INDEX][NAV_VELOCITY_EAST_INDEX] = dt_s;

    for (row = 0U; row < NAV_KALMAN_STATE_SIZE; ++row)
    {
        for (column = 0U; column < NAV_KALMAN_STATE_SIZE; ++column)
        {
            for (index = 0U; index < NAV_KALMAN_STATE_SIZE; ++index)
            {
                temporary[row][column]
                    += transition[row][index] * filter->covariance[index][column];
            }
        }
    }

    for (row = 0U; row < NAV_KALMAN_STATE_SIZE; ++row)
    {
        for (column = 0U; column < NAV_KALMAN_STATE_SIZE; ++column)
        {
            for (index = 0U; index < NAV_KALMAN_STATE_SIZE; ++index)
            {
                predicted_covariance[row][column]
                    += temporary[row][index] * transition[column][index];
            }
        }
    }

    acceleration_variance
        = NavKalman_Square(filter->config.acceleration_noise_std_mps2);

    predicted_covariance[NAV_NORTH_INDEX][NAV_NORTH_INDEX]
        += 0.25f * dt4 * acceleration_variance;
    predicted_covariance[NAV_NORTH_INDEX][NAV_VELOCITY_NORTH_INDEX]
        += 0.5f * dt3 * acceleration_variance;
    predicted_covariance[NAV_VELOCITY_NORTH_INDEX][NAV_NORTH_INDEX]
        += 0.5f * dt3 * acceleration_variance;
    predicted_covariance[NAV_VELOCITY_NORTH_INDEX][NAV_VELOCITY_NORTH_INDEX]
        += dt2 * acceleration_variance;

    predicted_covariance[NAV_EAST_INDEX][NAV_EAST_INDEX]
        += 0.25f * dt4 * acceleration_variance;
    predicted_covariance[NAV_EAST_INDEX][NAV_VELOCITY_EAST_INDEX]
        += 0.5f * dt3 * acceleration_variance;
    predicted_covariance[NAV_VELOCITY_EAST_INDEX][NAV_EAST_INDEX]
        += 0.5f * dt3 * acceleration_variance;
    predicted_covariance[NAV_VELOCITY_EAST_INDEX][NAV_VELOCITY_EAST_INDEX]
        += dt2 * acceleration_variance;

    memcpy(filter->covariance, predicted_covariance, sizeof(predicted_covariance));
    NavKalman_SymmetrizeCovariance(filter);
    return true;
}

bool NavKalman_UpdateGps(NavKalman_t *filter,
                         float north_m,
                         float east_m,
                         float velocity_north_mps,
                         float velocity_east_mps,
                         bool velocity_valid,
                         float position_std_m,
                         float velocity_std_mps)
{
    float position_variance;
    float velocity_variance;
    float innovation_north;
    float innovation_east;
    float s00;
    float s01;
    float s10;
    float s11;
    float determinant;
    float nis;

    if ((filter == NULL) || !isfinite(north_m) || !isfinite(east_m))
    {
        return false;
    }

    if (!filter->initialized)
    {
        NavKalman_SetState(filter,
                           north_m,
                           east_m,
                           (velocity_valid && isfinite(velocity_north_mps))
                               ? velocity_north_mps : 0.0f,
                           (velocity_valid && isfinite(velocity_east_mps))
                               ? velocity_east_mps : 0.0f);
        filter->accepted_gps_count++;
        filter->last_position_nis = 0.0f;
        return true;
    }

    if (!isfinite(position_std_m) || (position_std_m <= 0.0f))
    {
        position_std_m = filter->config.default_gps_position_std_m;
    }
    if (!isfinite(velocity_std_mps) || (velocity_std_mps <= 0.0f))
    {
        velocity_std_mps = filter->config.default_gps_velocity_std_mps;
    }

    position_variance = NavKalman_Square(position_std_m);
    velocity_variance = NavKalman_Square(velocity_std_mps);

    innovation_north = north_m - filter->state[NAV_NORTH_INDEX];
    innovation_east = east_m - filter->state[NAV_EAST_INDEX];
    s00 = filter->covariance[NAV_NORTH_INDEX][NAV_NORTH_INDEX]
        + position_variance;
    s01 = filter->covariance[NAV_NORTH_INDEX][NAV_EAST_INDEX];
    s10 = filter->covariance[NAV_EAST_INDEX][NAV_NORTH_INDEX];
    s11 = filter->covariance[NAV_EAST_INDEX][NAV_EAST_INDEX]
        + position_variance;
    determinant = (s00 * s11) - (s01 * s10);

    if (determinant <= NAV_MIN_VARIANCE)
    {
        filter->rejected_gps_count++;
        return false;
    }

    nis = (innovation_north * ((s11 * innovation_north)
                              - (s01 * innovation_east))
         + innovation_east * ((-s10 * innovation_north)
                              + (s00 * innovation_east))) / determinant;
    filter->last_position_nis = nis;

    if ((filter->config.position_gate_threshold > 0.0f)
            && (nis > filter->config.position_gate_threshold))
    {
        filter->rejected_gps_count++;
        return false;
    }

    (void)NavKalman_ScalarUpdate(filter, NAV_NORTH_INDEX,
                                 north_m, position_variance);
    (void)NavKalman_ScalarUpdate(filter, NAV_EAST_INDEX,
                                 east_m, position_variance);

    if (velocity_valid && isfinite(velocity_north_mps)
            && isfinite(velocity_east_mps))
    {
        (void)NavKalman_ScalarUpdate(filter, NAV_VELOCITY_NORTH_INDEX,
                                     velocity_north_mps, velocity_variance);
        (void)NavKalman_ScalarUpdate(filter, NAV_VELOCITY_EAST_INDEX,
                                     velocity_east_mps, velocity_variance);
    }

    filter->accepted_gps_count++;
    return true;
}

void NavKalman_GetState(const NavKalman_t *filter, NavKalman_State_t *state)
{
    if ((filter == NULL) || (state == NULL))
    {
        return;
    }

    state->north_m = filter->state[NAV_NORTH_INDEX];
    state->east_m = filter->state[NAV_EAST_INDEX];
    state->velocity_north_mps = filter->state[NAV_VELOCITY_NORTH_INDEX];
    state->velocity_east_mps = filter->state[NAV_VELOCITY_EAST_INDEX];
}
