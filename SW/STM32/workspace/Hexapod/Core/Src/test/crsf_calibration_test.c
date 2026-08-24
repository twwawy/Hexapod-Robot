#include "test/crsf_calibration_test.h"

#include <stddef.h>
#include <string.h>

/* 실제 CRSF 채널 범위 기록을 초기화한다. */
void CrsfCalibrationTest_Init(CrsfCalibrationTest_t *test)
{
    uint32_t channel;  // 초기화할 채널 번호를 저장한다.

    if (test == NULL)
    {
        return;
    }

    memset(test, 0, sizeof(*test));  // 이전 채널 기록을 제거한다.
    for (channel = 0U; channel < USER_COMMAND_USED_CHANNELS; ++channel)
    {
        test->minimum[channel] = UINT16_MAX;  // 첫 raw를 최소값으로 받을 준비를 한다.
    }
}

/* CH1~CH10의 실제 최소·최대 raw를 갱신한다. */
void CrsfCalibrationTest_Update(CrsfCalibrationTest_t *test,
                                const uint16_t raw[ROBOT_CRSF_CHANNEL_COUNT])
{
    uint32_t channel;  // 측정할 채널 번호를 저장한다.

    if ((test == NULL) || (raw == NULL))
    {
        return;
    }

    for (channel = 0U; channel < USER_COMMAND_USED_CHANNELS; ++channel)
    {
        if (raw[channel] < test->minimum[channel])
        {
            test->minimum[channel] = raw[channel];  // 새 최소 raw를 저장한다.
        }
        if (raw[channel] > test->maximum[channel])
        {
            test->maximum[channel] = raw[channel];  // 새 최대 raw를 저장한다.
        }
    }
}

/* 네 짐벌 중립과 스위치 가운데 위치의 실제 raw를 저장한다. */
void CrsfCalibrationTest_CaptureCenter(CrsfCalibrationTest_t *test,
                                       const uint16_t raw[ROBOT_CRSF_CHANNEL_COUNT])
{
    if ((test == NULL) || (raw == NULL))
    {
        return;
    }

    memcpy(test->center, raw, sizeof(test->center));  // CH1~CH10 현재 raw를 복사한다.
    test->center_captured = true;                     // 중심 위치 측정을 표시한다.
}

/* 관측 범위와 방향으로 UserCommand 보정표를 만든다. */
bool CrsfCalibrationTest_Build(const CrsfCalibrationTest_t *test,
                               const int8_t direction[USER_COMMAND_USED_CHANNELS],
                               UserCommand_ChannelCalibration_t table[USER_COMMAND_USED_CHANNELS])
{
    uint32_t channel;  // 보정할 채널 번호를 저장한다.

    if ((test == NULL) || (direction == NULL) || (table == NULL) ||
        !test->center_captured)
    {
        return false;
    }

    for (channel = 0U; channel < USER_COMMAND_USED_CHANNELS; ++channel)
    {
        const uint16_t center = (channel < 4U) ? test->center[channel] :
            (uint16_t)(((uint32_t)test->minimum[channel] + test->maximum[channel]) / 2U);  // 스위치는 관측 범위 중간을 경계 기준으로 둔다.

        if ((test->minimum[channel] >= center) ||
            (center >= test->maximum[channel]) ||
            ((direction[channel] != 1) && (direction[channel] != -1)))
        {
            return false;
        }

        table[channel].raw_min = test->minimum[channel];        // 실측 최소 raw를 저장한다.
        table[channel].raw_center = center;                     // 짐벌 중립 또는 스위치 경계를 저장한다.
        table[channel].raw_max = test->maximum[channel];        // 실측 최대 raw를 저장한다.
        table[channel].direction = direction[channel];         // 실측 방향을 저장한다.
        table[channel].switch_map[0] = 0U;                      // Low 논리값을 저장한다.
        table[channel].switch_map[1] = 1U;                      // Mid 논리값을 저장한다.
        table[channel].switch_map[2] = 2U;                      // High 논리값을 저장한다.
        table[channel].calibrated = true;                       // 실측 완료를 표시한다.
    }

    return true;
}
