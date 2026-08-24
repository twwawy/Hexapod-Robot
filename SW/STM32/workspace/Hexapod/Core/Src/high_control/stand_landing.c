#include "high_control/stand_landing.h"

#include <math.h>
#include <stddef.h>

#define STAND_FOOT_LIFT_M      0.06f
#define STAND_PARTIAL_HEIGHT_M 0.20f

/* 지정 구간에 Quintic 시간 스케일링을 적용한다. */
static float StandLanding_QuinticInterval(float input, float start, float end)
{
    float progress = (input - start) / (end - start);  // 구간 진행률을 계산한다.
    progress = fminf(fmaxf(progress, 0.0f), 1.0f);     // 진행률을 0~1로 제한한다.
    return 10.0f * progress * progress * progress -
           15.0f * progress * progress * progress * progress +
           6.0f * progress * progress * progress * progress * progress;  // Quintic 결과를 반환한다.
}

/* 서기·착지 진행률을 여섯 발의 Body 변화량으로 변환한다. */
void StandLanding_Calculate(bool stand_enable,
                            bool landing_enable,
                            float posture_progress,
                            RobotVec3_t delta_body[ROBOT_LEG_COUNT])
{
    static const float alpha[ROBOT_LEG_COUNT] =
    {
        -45.0f * ROBOT_DEG_TO_RAD_F, -90.0f * ROBOT_DEG_TO_RAD_F,
        -135.0f * ROBOT_DEG_TO_RAD_F, 45.0f * ROBOT_DEG_TO_RAD_F,
        90.0f * ROBOT_DEG_TO_RAD_F, 135.0f * ROBOT_DEG_TO_RAD_F
    };  // 여섯 다리 장착각을 저장한다.
    const float q2_initial = 30.0f * ROBOT_DEG_TO_RAD_F;  // 기본 Femur 각을 저장한다.
    const float q3_initial = 50.0f * ROBOT_DEG_TO_RAD_F;  // 기본 Tibia 각을 저장한다.
    const float q3_middle = 100.0f * ROBOT_DEG_TO_RAD_F;  // 중간 Tibia 각을 저장한다.
    const float nominal_radial = ROBOT_LINK_1_M + ROBOT_LINK_2_M * cosf(q2_initial) +
                                 ROBOT_LINK_3_M * cosf(q2_initial + q3_initial);  // 기본 반경을 계산한다.
    const float nominal_z = -ROBOT_LINK_2_M * sinf(q2_initial) -
                            ROBOT_LINK_3_M * sinf(q2_initial + q3_initial);       // 기본 높이를 계산한다.
    const float landed_radial = ROBOT_LINK_1_M + ROBOT_LINK_2_M + ROBOT_LINK_3_M;  // 착지 반경을 계산한다.
    const float middle_link = sqrtf(ROBOT_LINK_2_M * ROBOT_LINK_2_M +
                                    ROBOT_LINK_3_M * ROBOT_LINK_3_M +
                                    2.0f * ROBOT_LINK_2_M * ROBOT_LINK_3_M * cosf(q3_middle));  // 중간 링크 길이를 계산한다.
    const float middle_radial = ROBOT_LINK_1_M + middle_link;  // 중간 반경을 계산한다.
    const float progress = fminf(fmaxf(posture_progress, 0.0f), 1.0f);  // 자세 진행률을 제한한다.
    const float first_135 = StandLanding_QuinticInterval(progress, 0.0f, 1.0f / 7.0f);      // 1·3·5 첫 이동률을 계산한다.
    const float first_246 = StandLanding_QuinticInterval(progress, 1.0f / 7.0f, 2.0f / 7.0f);// 2·4·6 첫 이동률을 계산한다.
    const float raise_first = StandLanding_QuinticInterval(progress, 2.0f / 7.0f, 3.5f / 7.0f);// 첫 몸체 상승률을 계산한다.
    const float second_135 = StandLanding_QuinticInterval(progress, 3.5f / 7.0f, 4.5f / 7.0f);// 1·3·5 둘째 이동률을 계산한다.
    const float second_246 = StandLanding_QuinticInterval(progress, 4.5f / 7.0f, 5.5f / 7.0f);// 2·4·6 둘째 이동률을 계산한다.
    const float raise_second = StandLanding_QuinticInterval(progress, 5.5f / 7.0f, 1.0f);    // 둘째 몸체 상승률을 계산한다.
    const float radial_135 = landed_radial + first_135 * (middle_radial - landed_radial) +
                             second_135 * (nominal_radial - middle_radial);  // 1·3·5 반경을 계산한다.
    const float radial_246 = landed_radial + first_246 * (middle_radial - landed_radial) +
                             second_246 * (nominal_radial - middle_radial);  // 2·4·6 반경을 계산한다.
    const float support_z = raise_first * (-STAND_PARTIAL_HEIGHT_M) +
                            raise_second * (nominal_z + STAND_PARTIAL_HEIGHT_M);  // 지지 높이를 계산한다.
    const float lift_135 = STAND_FOOT_LIFT_M *
                           (4.0f * first_135 * (1.0f - first_135) +
                            4.0f * second_135 * (1.0f - second_135));  // 1·3·5 발 들림을 계산한다.
    const float lift_246 = STAND_FOOT_LIFT_M *
                           (4.0f * first_246 * (1.0f - first_246) +
                            4.0f * second_246 * (1.0f - second_246));  // 2·4·6 발 들림을 계산한다.
    uint32_t leg;  // 출력할 다리 번호를 저장한다.

    if (delta_body == NULL)
    {
        return;
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        const bool group_135 = ((leg % 2U) == 0U);  // 1·3·5 그룹 여부를 계산한다.
        const float radial_delta = (group_135 ? radial_135 : radial_246) - nominal_radial;  // 반경 변화량을 계산한다.
        const float vertical_delta = support_z + (group_135 ? lift_135 : lift_246) - nominal_z;  // 높이 변화량을 계산한다.

        if (stand_enable || landing_enable)
        {
            delta_body[leg].x = radial_delta * cosf(alpha[leg]);  // Body X 변화량을 계산한다.
            delta_body[leg].y = radial_delta * sinf(alpha[leg]);  // Body Y 변화량을 계산한다.
            delta_body[leg].z = vertical_delta;                   // Body Z 변화량을 계산한다.
        }
        else
        {
            delta_body[leg].x = 0.0f;  // 비활성 X 변화량을 제거한다.
            delta_body[leg].y = 0.0f;  // 비활성 Y 변화량을 제거한다.
            delta_body[leg].z = 0.0f;  // 비활성 Z 변화량을 제거한다.
        }
    }
}
