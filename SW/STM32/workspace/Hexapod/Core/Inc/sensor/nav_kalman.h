#ifndef NAV_KALMAN_H
#define NAV_KALMAN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

#define NAV_KALMAN_STATE_SIZE 4U

typedef struct
{
    float acceleration_noise_std_mps2;
    float initial_position_std_m;
    float initial_velocity_std_mps;
    float default_gps_position_std_m;
    float default_gps_velocity_std_mps;
    float position_gate_threshold;
} NavKalman_Config_t;

typedef struct
{
    float north_m;
    float east_m;
    float velocity_north_mps;
    float velocity_east_mps;
} NavKalman_State_t;

typedef struct
{
    float state[NAV_KALMAN_STATE_SIZE];
    float covariance[NAV_KALMAN_STATE_SIZE][NAV_KALMAN_STATE_SIZE];
    NavKalman_Config_t config;

    bool initialized;
    float last_position_nis;
    uint32_t accepted_gps_count;
    uint32_t rejected_gps_count;
} NavKalman_t;

/** Load safe starting values for a planar GPS/IMU filter. */
void NavKalman_DefaultConfig(NavKalman_Config_t *config);

/** Initialize a 4-state [N, E, Vn, Ve] Kalman filter. */
void NavKalman_Init(NavKalman_t *filter, const NavKalman_Config_t *config);

/** Set the initial local position and velocity. */
void NavKalman_SetState(NavKalman_t *filter,
                        float north_m,
                        float east_m,
                        float velocity_north_mps,
                        float velocity_east_mps);

/**
 * @brief Predict from navigation-frame acceleration.
 * @param acceleration_north_mps2 Linear acceleration with gravity removed.
 * @param acceleration_east_mps2  Linear acceleration with gravity removed.
 * @note  Do not pass raw body-frame WT931 acceleration directly. Rotate it to
 *        the navigation frame and remove gravity first.
 */
bool NavKalman_Predict(NavKalman_t *filter,
                       float acceleration_north_mps2,
                       float acceleration_east_mps2,
                       float dt_s);

/**
 * @brief Correct position and, when available, velocity with GPS.
 * @return true when the measurement was accepted by the innovation gate.
 * @note  Pass a non-positive std value to use the configured default.
 */
bool NavKalman_UpdateGps(NavKalman_t *filter,
                         float north_m,
                         float east_m,
                         float velocity_north_mps,
                         float velocity_east_mps,
                         bool velocity_valid,
                         float position_std_m,
                         float velocity_std_mps);

void NavKalman_GetState(const NavKalman_t *filter, NavKalman_State_t *state);

#ifdef __cplusplus
}
#endif

#endif /* NAV_KALMAN_H */
