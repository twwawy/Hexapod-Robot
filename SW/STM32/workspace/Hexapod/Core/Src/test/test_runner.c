#include "test/test_runner.h"

#include <stddef.h>
#include <string.h>

/* 선택한 테스트 하나를 초기화하고 실행 상태로 만든다. */
bool TestRunner_Start(TestRunner_Handle_t *handle,
                      TestRunner_InitFunction_t initialize,
                      TestRunner_ProcessFunction_t process,
                      void *context)
{
    if ((handle == NULL) || (initialize == NULL) || (process == NULL))
    {
        return false;
    }

    memset(handle, 0, sizeof(*handle));  // 이전 테스트 상태를 제거한다.
    handle->initialize = initialize;     // 초기화 함수를 연결한다.
    handle->process = process;           // 반복 함수를 연결한다.
    handle->context = context;           // 테스트별 상태를 연결한다.

    if (!handle->initialize(handle->context))
    {
        handle->state = TEST_RUNNER_FAILED;  // 초기화 실패를 기록한다.
        return false;
    }

    handle->state = TEST_RUNNER_RUNNING;  // 반복 실행을 허가한다.
    return true;
}

/* 현재 테스트를 한 번 실행하고 완료 여부를 갱신한다. */
TestRunner_State_t TestRunner_Process(TestRunner_Handle_t *handle)
{
    if ((handle == NULL) || (handle->state != TEST_RUNNER_RUNNING))
    {
        return (handle == NULL) ? TEST_RUNNER_FAILED : handle->state;
    }

    handle->process_count++;  // 실행 횟수를 기록한다.
    if (handle->process(handle->context))
    {
        handle->state = TEST_RUNNER_PASSED;  // 테스트 자체 완료를 기록한다.
    }

    return handle->state;
}

/* 현재 테스트를 즉시 실패 상태로 만든다. */
void TestRunner_Fail(TestRunner_Handle_t *handle)
{
    if (handle != NULL)
    {
        handle->state = TEST_RUNNER_FAILED;  // 실패 상태를 유지한다.
    }
}
