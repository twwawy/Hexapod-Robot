#ifndef RL_CONTROLLER_H
#define RL_CONTROLLER_H

#include "common/robot_types.h"

typedef enum
{
    RL_SUBMIT_ACCEPTED = 0,      // 검증한 새 출력을 수락한다.
    RL_SUBMIT_INVALID_ARGUMENT,  // 필수 입력 누락을 나타낸다.
    RL_SUBMIT_INACTIVE,          // 비활성 세션 입력을 나타낸다.
    RL_SUBMIT_SESSION_MISMATCH,  // 이전 운용 세션 입력을 나타낸다.
    RL_SUBMIT_SEQUENCE,          // 중복 또는 역순 출력을 나타낸다.
    RL_SUBMIT_OBSERVATION,       // 누락되거나 오래된 관측을 나타낸다.
    RL_SUBMIT_PLAN,              // 광고한 보행 계획의 불일치를 나타낸다.
    RL_SUBMIT_VALUE              // 자세 또는 잔차 범위 오류를 나타낸다.
} RlController_SubmitResult_t;

typedef struct
{
    uint32_t timestamp_ms;  // 관측을 공개한 로컬 시각을 저장한다.
    uint16_t sequence;      // 공개한 관측 순번을 저장한다.
    uint16_t plan_id;       // 관측에 포함한 보행 계획을 저장한다.
    uint8_t swing_mask;     // 관측에 포함한 이륙 다리를 저장한다.
    bool plan_valid;        // 관측의 다리 계획 유효성을 저장한다.
    bool valid;             // 관측 슬롯 사용 여부를 저장한다.
} RlController_Observation_t;

typedef struct
{
    RobotRlAction_t action;          // 마지막으로 수락한 출력을 저장한다.
    uint32_t session_id;             // 현재 운용 세션 번호를 저장한다.
    uint32_t last_action_ms;         // 마지막 출력 수락 시각을 저장한다.
    uint32_t action_observation_ms;  // 수락한 출력의 원관측 시각을 저장한다.
    uint32_t accepted_count;         // 현재 세션의 수락 횟수를 저장한다.
    uint32_t rejected_count;         // 현재 세션의 거부 횟수를 저장한다.

    uint16_t plan_id;                      // 현재 광고한 보행 계획을 저장한다.
    uint16_t latest_observation_sequence;  // 관측 재전송 검사용 순번을 저장한다.
    uint8_t swing_mask;                    // 현재 광고한 이륙 다리를 저장한다.
    uint8_t observation_write_index;       // 다음 관측 저장 위치를 저장한다.

    bool session_active;       // 운용 세션 활성 여부를 저장한다.
    bool plan_valid;           // 현재 다리 계획 유효성을 저장한다.
    bool action_valid;         // 마지막 출력의 검증 완료를 저장한다.
    bool has_received_action;  // 첫 출력 대기와 시간 초과를 구분한다.
    bool has_observation;      // 관측 순번 비교 가능 여부를 저장한다.

    RlController_SubmitResult_t last_result;  // 마지막 제출 결과를 저장한다.

    RlController_Observation_t observation[ROBOT_RL_OBSERVATION_HISTORY_COUNT];  // 최근 관측의 시각과 계획을 저장한다.
} RlController_Handle_t;

void RlController_Init(RlController_Handle_t *handle);  // 운용 세션과 수신 이력을 초기화한다.

void RlController_BeginSession(RlController_Handle_t *handle, uint32_t session_id);  // 외부에서 발급한 새 세션을 시작한다.

void RlController_EndSession(RlController_Handle_t *handle);  // 기존 세션의 관측과 출력을 폐기한다.

void RlController_SetPlan(RlController_Handle_t *handle,
                          uint16_t plan_id, uint8_t swing_mask, bool valid);  // 다음 출력에 요구할 계획을 설정한다.

void RlController_RecordObservation(RlController_Handle_t *handle,
                                    uint16_t sequence, uint32_t now_ms);  // 실제 공개한 관측 순번과 시각을 기록한다.

RlController_SubmitResult_t RlController_Submit(RlController_Handle_t *handle,
                                               const RobotRlAction_t *action,
                                               uint32_t now_ms);  // 세션·관측·계획·수치 범위를 검증해 출력을 수락한다.

bool RlController_GetAction(const RlController_Handle_t *handle,
                            RobotRlAction_t *action, uint32_t now_ms);  // 신선한 출력만 제어 주기에 복사한다.

#endif
