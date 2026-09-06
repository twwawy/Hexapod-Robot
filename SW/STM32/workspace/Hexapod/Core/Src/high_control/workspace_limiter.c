#include "high_control/workspace_limiter.h"

#include "high_control/leg_kinematics.h"
#include "high_control/stance_trajectory.h"
#include "high_control/swing_trajectory.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define WORKSPACE_GAIT_COMMAND_HOLD_STEPS 2U     // 한 조종 명령을 유지할 걸음 수를 정의한다.
#define WORKSPACE_GAIT_SCALE_REDUCTION    0.8f   // 실패한 보폭을 줄일 공통 비율을 정의한다.
#define WORKSPACE_GAIT_MAX_REDUCTIONS     8U     // 불가능한 경로의 축소 재시도 횟수를 제한한다.

static const float workspace_leg_angle_rad[ROBOT_LEG_COUNT] =
{
    -45.0f * ROBOT_DEG_TO_RAD_F,   // 1번 다리 장착각을 저장한다.
    -90.0f * ROBOT_DEG_TO_RAD_F,   // 2번 다리 장착각을 저장한다.
    -135.0f * ROBOT_DEG_TO_RAD_F,  // 3번 다리 장착각을 저장한다.
    45.0f * ROBOT_DEG_TO_RAD_F,    // 4번 다리 장착각을 저장한다.
    90.0f * ROBOT_DEG_TO_RAD_F,    // 5번 다리 장착각을 저장한다.
    135.0f * ROBOT_DEG_TO_RAD_F    // 6번 다리 장착각을 저장한다.
};

/* Body 발 벡터에 자세의 역회전을 적용한다. */
static RobotVec3_t WorkspaceLimiter_RotateInverse(const RobotVec3_t *input,
                                                  const RobotEuler_t *posture)
{
    RobotVec3_t output;                      // 역회전한 발 위치를 저장한다.
    const float cr = cosf(posture->roll);   // Roll Cosine을 계산한다.
    const float sr = sinf(posture->roll);   // Roll Sine을 계산한다.
    const float cp = cosf(posture->pitch);  // Pitch Cosine을 계산한다.
    const float sp = sinf(posture->pitch);  // Pitch Sine을 계산한다.
    const float cy = cosf(posture->yaw);    // Yaw Cosine을 계산한다.
    const float sy = sinf(posture->yaw);    // Yaw Sine을 계산한다.
    const float r11 = cy * cp;              // 회전행렬 1행 1열을 계산한다.
    const float r12 = cy * sp * sr - sy * cr;
    const float r13 = cy * sp * cr + sy * sr;
    const float r21 = sy * cp;
    const float r22 = sy * sp * sr + cy * cr;
    const float r23 = sy * sp * cr - cy * sr;
    const float r31 = -sp;
    const float r32 = cp * sr;
    const float r33 = cp * cr;

    output.x = r11 * input->x + r21 * input->y + r31 * input->z;  // 역회전 X를 계산한다.
    output.y = r12 * input->x + r22 * input->y + r32 * input->z;  // 역회전 Y를 계산한다.
    output.z = r13 * input->x + r23 * input->y + r33 * input->z;  // 역회전 Z를 계산한다.
    return output;
}

/* 두 보행 명령이 같은지 확인한다. */
static bool WorkspaceLimiter_TwistEqual(const RobotBodyTwist_t *first,
                                        const RobotBodyTwist_t *second)
{
    return (first->vx == second->vx) &&
           (first->vy == second->vy) &&
           (first->vz == second->vz) &&
           (first->wz == second->wz);  // 네 축 명령을 함께 비교한다.
}

/* 작은 입력·위치 보정 변화에는 기존 보폭 비율을 재사용한다. */
static bool WorkspaceLimiter_TwistNearby(const RobotBodyTwist_t *first,
                                         const RobotBodyTwist_t *second)
{
    return (fabsf(first->vx - second->vx) <= ROBOT_GAIT_LINEAR_THRESHOLD_MPS) &&
           (fabsf(first->vy - second->vy) <= ROBOT_GAIT_LINEAR_THRESHOLD_MPS) &&
           (fabsf(first->vz - second->vz) <= ROBOT_GAIT_LINEAR_THRESHOLD_MPS) &&
           (fabsf(first->wz - second->wz) <= ROBOT_GAIT_YAW_THRESHOLD_RADPS);  // 보행 판정 폭 안의 입력 흔들림을 허용한다.
}

/* 이동 방향과 회전 비율을 유지하며 보폭을 줄인다. */
static RobotBodyTwist_t WorkspaceLimiter_ScaleTwist(const RobotBodyTwist_t *input,
                                                    float scale)
{
    RobotBodyTwist_t output;  // 공통 비율을 적용한 명령을 저장한다.

    output.vx = input->vx * scale;  // 전후 보폭을 축소한다.
    output.vy = input->vy * scale;  // 좌우 보폭을 축소한다.
    output.vz = input->vz * scale;  // 수직 보폭을 축소한다.
    output.wz = input->wz * scale;  // 이동에 대한 회전 비율을 유지한다.
    return output;
}

/* 사용자 회전 명령에 최신 Heading 보정을 합친다. */
static RobotBodyTwist_t WorkspaceLimiter_ComposeYaw(const RobotBodyTwist_t *user_command,
                                                     float yaw_feedback_radps)
{
    RobotBodyTwist_t output = *user_command;  // 두 걸음용 사용자 명령을 복사한다.

    output.wz = fminf(fmaxf(user_command->wz + yaw_feedback_radps,
                            -ROBOT_MAX_YAW_RATE_RADPS),
                      ROBOT_MAX_YAW_RATE_RADPS);  // 최신 Heading 보정을 제한해 합친다.
    return output;
}

/* 새 보행 명령의 세 지점 검사를 시작한다. */
static void WorkspaceLimiter_StartPreview(WorkspaceLimiter_Handle_t *handle,
                                           const RobotBodyTwist_t *user_candidate,
                                           const RobotGaitPhase_t *gait)
{
    const bool reuse_scale = (handle->gait_applied_scale > 0.0f) &&
        (handle->applied_pattern == gait->next_phase_pattern) &&
        WorkspaceLimiter_TwistNearby(user_candidate,
                                      &handle->gait_requested);  // 작은 보정 변화에도 기존 축소 비율을 찾는다.

    handle->gait_pending = *user_candidate;                    // 사용자 후보를 검사 동안 고정한다.
    handle->preview_scale = reuse_scale ?
                            handle->gait_applied_scale : 1.0f;  // 같은 입력의 반복적인 보폭 탐색을 막는다.
    handle->preview_reduction_count = 0U;                       // 새 위상의 축소 횟수를 초기화한다.
    handle->preview_sample = 0U;                               // 첫 경로 지점부터 시작한다.
    handle->preview_swing_mask = gait->next_phase_swing_mask;  // 다음 Tripod 역할을 고정한다.
    handle->preview_startup_phase = gait->next_phase_startup;  // 첫 위상 여부를 저장한다.
    handle->preview_continuous = (gait->next_phase_pattern == ROBOT_GAIT_WAVE) ||
                                (handle->applied_pattern != gait->next_phase_pattern);  // 패턴 교체도 현재 발 위치에서 검사한다.
    handle->preview_pattern = gait->next_phase_pattern;       // 착지 후 적용할 패턴을 고정한다.
    memcpy(handle->preview_feet, handle->current_feet,
           sizeof(handle->preview_feet));                     // 검사 중 발 시작점 변경을 막는다.
    handle->preview_offset_m = handle->body_offset_m;         // 보정된 기준 위치를 고정한다.
    handle->preview_active = true;                             // 세 지점 검사를 활성화한다.
    handle->phase_result_valid = false;                        // 이전 위상 검사 결과를 제거한다.
    handle->phase_result_accepted = false;                     // 새 후보를 미통과 상태로 둔다.
}

/* 한 검사 지점의 실제 Tripod 발 목표 여섯 개를 함께 검사한다. */
static bool WorkspaceLimiter_PreviewPoint(const WorkspaceLimiter_Handle_t *handle,
                                          const RobotEuler_t *posture)
{
    RobotVec3_t base[ROBOT_LEG_COUNT];  // 기본 발 위치를 저장한다.
    RobotVec3_t feet[ROBOT_LEG_COUNT];  // 검사할 여섯 발 위치를 저장한다.
    const float progress =
        (float)handle->preview_sample /
        (float)(ROBOT_GAIT_PREVIEW_SAMPLE_COUNT - 1U);  // 시작·중앙·끝 세 지점을 만든다.
    uint32_t leg;  // 계산할 다리 번호를 저장한다.

    LegKinematics_GetBaseFeet(base);  // 기본 발 위치를 계산한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        RobotVec3_t displacement;  // 후보 명령의 한 위상 이동량을 저장한다.
        RobotVec3_t front;         // Swing 착지점을 저장한다.
        RobotVec3_t rear;          // Swing 시작점을 저장한다.
        const bool swing = (handle->preview_swing_mask &
                            (uint8_t)(1U << leg)) != 0U;  // 현재 다리 역할을 선택한다.

        if (handle->preview_continuous)
        {
            base[leg].x -= handle->preview_offset_m.x;  // 실제 궤적의 기준 X를 맞춘다.
            base[leg].y -= handle->preview_offset_m.y;  // 실제 궤적의 기준 Y를 맞춘다.
            base[leg].z -= handle->preview_offset_m.z;  // 실제 궤적의 기준 Z를 맞춘다.
        }

        displacement.x = ROBOT_GAIT_PHASE_TIME_S *
                         (-handle->gait_preview.vx +
                          handle->gait_preview.wz * base[leg].y);  // Stance X 이동량을 계산한다.
        displacement.y = ROBOT_GAIT_PHASE_TIME_S *
                         (-handle->gait_preview.vy -
                          handle->gait_preview.wz * base[leg].x);  // Stance Y 이동량을 계산한다.
        displacement.z = ROBOT_GAIT_PHASE_TIME_S *
                         (-handle->gait_preview.vz);               // Stance Z 이동량을 계산한다.
        const float stance_phases = (handle->preview_pattern == ROBOT_GAIT_WAVE)
                                  ? (float)ROBOT_WAVE_STANCE_PHASES : 1.0f;  // 전체 지지 구간을 선택한다.

        front.x = base[leg].x - 0.5f * stance_phases * displacement.x;  // 앞쪽 X 끝점을 계산한다.
        front.y = base[leg].y - 0.5f * stance_phases * displacement.y;  // 앞쪽 Y 끝점을 계산한다.
        front.z = base[leg].z - 0.5f * stance_phases * displacement.z;  // 앞쪽 Z 끝점을 계산한다.
        rear.x = base[leg].x + 0.5f * displacement.x;              // 뒤쪽 X 끝점을 계산한다.
        rear.y = base[leg].y + 0.5f * displacement.y;              // 뒤쪽 Y 끝점을 계산한다.
        rear.z = base[leg].z + 0.5f * displacement.z;              // 뒤쪽 Z 끝점을 계산한다.

        if (handle->preview_continuous)
        {
            if (swing)
            {
                const float height = fminf(fmaxf(ROBOT_SWING_HEIGHT_M + handle->preview_offset_m.z,
                                                  ROBOT_SWING_HEIGHT_MIN_M),
                                            ROBOT_SWING_HEIGHT_MAX_M);  // 실제 보정 높이와 같은 Swing을 검사한다.

                feet[leg] = SwingTrajectory_Calculate(progress, &handle->preview_feet[leg],
                                                       &front, height, ROBOT_SWING_RADIAL_OFFSET_M,
                                                       workspace_leg_angle_rad[leg]);  // 실제 이륙 위치부터 단일 발 경로를 검사한다.
            }
            else
            {
                const float duration_s = progress * ROBOT_GAIT_PHASE_TIME_S *
                    ((handle->preview_startup_phase && (handle->preview_pattern == ROBOT_GAIT_WAVE))
                     ? 0.5f : 1.0f);  // 개별 보행 첫 순회의 지지 이동량을 맞춘다.

                feet[leg] = StanceTrajectory_Advance(&handle->preview_feet[leg],
                                                     &handle->gait_preview, duration_s);  // 다섯 지지점의 실제 다음 위치를 검사한다.
            }
        }
        else if (swing)
        {
            const RobotVec3_t *start = handle->preview_startup_phase
                                     ? &base[leg]
                                     : &rear;  // 첫 위상 Swing 시작점을 선택한다.
            feet[leg] = SwingTrajectory_Calculate(progress,
                                                   start,
                                                   &front,
                                                   ROBOT_SWING_HEIGHT_M,
                                                   ROBOT_SWING_RADIAL_OFFSET_M,
                                                   workspace_leg_angle_rad[leg]);  // Swing 목표를 계산한다.
        }
        else if (handle->preview_startup_phase)
        {
            feet[leg] = StanceTrajectory_Interpolate(progress,
                                                      &base[leg],
                                                      &rear);  // 첫 위상 Stance 목표를 계산한다.
        }
        else
        {
            feet[leg] = StanceTrajectory_Interpolate(progress,
                                                      &front,
                                                      &rear);  // 정상 Stance 목표를 계산한다.
        }
    }

    return WorkspaceLimiter_AllFeetValid(feet, posture);  // 여섯 IK를 한 번씩만 검사한다.
}

/* 마지막 적용 보행 명령을 0으로 초기화한다. */
void WorkspaceLimiter_Init(WorkspaceLimiter_Handle_t *handle)
{
    if (handle != NULL)
    {
        memset(handle, 0, sizeof(*handle));  // 이전 적용 명령을 제거한다.
        LegKinematics_GetBaseFeet(handle->current_feet);  // 단독 시험의 기본 시작점을 준비한다.
    }
}

/* 다음 위상 검사에 사용할 실제 발 목표와 몸체 보정을 전달한다. */
void WorkspaceLimiter_SetFeet(WorkspaceLimiter_Handle_t *handle,
                              const RobotVec3_t feet[ROBOT_LEG_COUNT],
                              const RobotVec3_t *body_offset_m)
{
    if ((handle == NULL) || (feet == NULL) || (body_offset_m == NULL))
    {
        return;
    }

    memcpy(handle->current_feet, feet, sizeof(handle->current_feet));  // 다음 검사의 실제 시작점을 갱신한다.
    handle->body_offset_m = *body_offset_m;                           // 보정된 궤적 기준점을 전달한다.
}

/* 자세 역회전 후 여섯 발의 IK 가능 여부를 검사한다. */
bool WorkspaceLimiter_AllFeetValid(const RobotVec3_t feet_body[ROBOT_LEG_COUNT],
                                   const RobotEuler_t *posture_rad)
{
    uint32_t leg;  // 검사할 다리 번호를 저장한다.

    if ((feet_body == NULL) || (posture_rad == NULL))
    {
        return false;
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        const RobotVec3_t rotated = WorkspaceLimiter_RotateInverse(&feet_body[leg], posture_rad);  // 자세 후보를 적용한다.

        if (!LegKinematics_IsReachable((uint8_t)leg, &rotated))
        {
            return false;  // 한 다리라도 IK 불가능하면 전체 후보를 거부한다.
        }
    }

    return true;
}

/* 다음 보폭을 검사하고 실패하면 다음 주기에 축소 후보를 검사한다. */
RobotBodyTwist_t WorkspaceLimiter_Gait(WorkspaceLimiter_Handle_t *handle,
                                       const RobotBodyTwist_t *user_candidate,
                                       float yaw_feedback_radps,
                                       bool manual_enable,
                                       const RobotGaitPhase_t *gait,
                                       const RobotEuler_t *posture_rad,
                                       bool reset_command,
                                       bool *accepted)
{
    RobotBodyTwist_t output;                            // 최종 보행 명령을 저장한다.
    RobotBodyTwist_t zero = {0.0f, 0.0f, 0.0f, 0.0f};  // 오류 시 0 명령을 준비한다.

    if ((handle == NULL) || (user_candidate == NULL) || (gait == NULL) ||
        (posture_rad == NULL) || (accepted == NULL))
    {
        return zero;
    }

    *accepted = WorkspaceLimiter_TwistEqual(user_candidate,
                                              &handle->gait_applied);  // 현재 입력의 적용 여부를 표시한다.

    if (reset_command)
    {
        memset(handle, 0, sizeof(*handle));  // Reset에서 보행 검사 상태를 제거한다.
        *accepted = true;                    // 0 명령 적용을 완료 상태로 표시한다.
        return zero;
    }
    else if (manual_enable)
    {
        if (!gait->enabled_internal && gait->waiting_start && !gait->next_phase_preview)
        {
            handle->preview_active = false;         // 패턴 전환으로 취소한 경로 검사를 제거한다.
            handle->phase_result_valid = false;     // 이전 패턴의 검사 결과 재사용을 막는다.
            handle->phase_result_accepted = false;  // 새 패턴 검사 전 이륙을 차단한다.
        }
        if (!gait->enabled_internal && !gait->next_phase_preview &&
            !handle->preview_active)
        {
            handle->gait_applied_step_count = 0U;  // 정지 뒤 새 입력을 첫 걸음부터 받는다.
        }

        if (gait->next_phase_preview)
        {
            handle->preview_reuses_applied =
                (gait->next_phase_pattern == ROBOT_GAIT_TRIPOD) &&
                (handle->applied_pattern == gait->next_phase_pattern) &&
                (handle->gait_applied_step_count > 0U) &&
                (handle->gait_applied_step_count <
                 WORKSPACE_GAIT_COMMAND_HOLD_STEPS);  // 둘째 걸음의 기존 속도 사용 여부를 결정한다.
            WorkspaceLimiter_StartPreview(
                handle,
                handle->preview_reuses_applied ? &handle->gait_requested : user_candidate,
                gait);  // 두 걸음 사용자 속도와 최신 Heading 보정을 검사한다.
            *accepted = false;  // 세 지점 통과 전까지 적용을 보류한다.
        }

        if (handle->preview_active)
        {
            const RobotBodyTwist_t scaled = WorkspaceLimiter_ScaleTwist(
                &handle->gait_pending, handle->preview_scale);  // 이번 주기에 검사할 보폭 하나를 만든다.

            handle->gait_preview = WorkspaceLimiter_ComposeYaw(
                &scaled, yaw_feedback_radps);  // 축소한 보폭에 현재 Heading 보정을 합친다.
            *accepted = false;                 // 전체 경로 통과 전에는 채택을 보류한다.
        }

        while (handle->preview_active &&
               (handle->preview_sample < ROBOT_GAIT_PREVIEW_SAMPLE_COUNT))
        {
            if (!WorkspaceLimiter_PreviewPoint(handle, posture_rad))
            {
                if (handle->preview_reduction_count < WORKSPACE_GAIT_MAX_REDUCTIONS)
                {
                    handle->preview_scale *= WORKSPACE_GAIT_SCALE_REDUCTION;  // 방향을 유지하며 다음 보폭을 줄인다.
                    handle->preview_reduction_count++;                        // 제한된 재시도 횟수를 누적한다.
                    handle->preview_sample = 0U;                              // 축소 경로도 시작부터 다시 검사한다.
                }
                else
                {
                    handle->preview_active = false;         // 축소로 해결할 수 없는 검사를 종료한다.
                    handle->preview_reuses_applied = false;  // 실패한 두 걸음 재사용 결정을 제거한다.
                    handle->phase_result_valid = true;      // 최종 실패만 보행 상태기에 전달한다.
                    handle->phase_result_accepted = false;  // 불가능한 다음 위상은 계속 차단한다.
                }
                break;  // 5 ms 주기에 후보 하나만 검사하고 다음 주기에 자동 재시도한다.
            }
            else
            {
                handle->preview_sample++;  // 같은 5 ms 주기의 다음 검사 지점으로 이동한다.
                *accepted = false;         // 전체 경로 통과 전까지 적용을 보류한다.
            }
        }

        if (handle->preview_active &&
            (handle->preview_sample >= ROBOT_GAIT_PREVIEW_SAMPLE_COUNT))
        {
            handle->gait_applied = WorkspaceLimiter_ScaleTwist(
                &handle->gait_pending, handle->preview_scale);  // 검사를 통과한 실제 보폭만 적용한다.
            handle->gait_requested = handle->gait_pending;       // 동일 입력의 축소 비율 재사용 기준을 저장한다.
            handle->gait_applied_scale = handle->preview_scale;  // 다음 걸음의 유효한 보폭 비율을 저장한다.
            handle->applied_pattern = handle->preview_pattern;  // 다른 패턴의 명령 재사용을 막는다.

            if (handle->preview_reuses_applied)
            {
                handle->gait_applied_step_count++;  // 반대 Tripod의 둘째 걸음을 기록한다.
            }
            else
            {
                handle->gait_applied_step_count = 1U;  // 새 속도의 첫 걸음을 기록한다.
            }
            handle->preview_active = false;               // 한 주기 검사를 완료한다.
            handle->preview_reuses_applied = false;        // 완료한 재사용 결정을 제거한다.
            handle->phase_result_valid = true;            // 검사 완료 결과를 확정한다.
            handle->phase_result_accepted = true;         // 다음 위상 진입을 허용한다.
            *accepted = WorkspaceLimiter_TwistEqual(user_candidate,
                                                     &handle->gait_applied);  // 현재 후보 채택 여부를 표시한다.
        }
    }
    else
    {
        handle->gait_applied = *user_candidate;  // 보정 모드는 발 Offset 단계에서 검사한다.
        handle->gait_applied_scale = 0.0f;       // 다른 모드에서 보행 축소 비율을 제거한다.
        handle->gait_applied_step_count = 0U;   // 정상 보행의 두 걸음 기억을 제거한다.
        handle->preview_active = false;         // 수동 보행 검사를 중단한다.
        handle->preview_reuses_applied = false; // 진행 중인 재사용 결정을 제거한다.
        handle->phase_result_valid = false;     // 수동 위상 검사 결과를 제거한다.
        handle->phase_result_accepted = false;  // 수동 위상 허가를 제거한다.
        *accepted = true;                       // 보정 명령은 발 Offset 단계에서 검사한다.
    }

    output = WorkspaceLimiter_ComposeYaw(&handle->gait_applied,
                                         yaw_feedback_radps);  // 걸음 중에도 최신 Heading 보정을 적용한다.
    return output;
}
