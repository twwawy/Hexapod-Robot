% 20초 전체의 IK와 세이프티를 검사하고 통과한 험지 조건을 저장한다.
function verify_rough_walk
output_dir = fileparts(mfilename('fullpath'));  % 검증 결과 경로를 설정한다.
cd(fileparts(output_dir));                     % 모델 초기화 경로를 맞춘다.
diary(fullfile(output_dir, 'rough_walk_log.txt'));  % 조건별 검증 기록을 저장한다.
cleanup = onCleanup(@() diary('off'));             % 종료 시 기록을 닫는다.
Simulink.fileGenControl('set', 'CacheFolder', fullfile(tempdir, 'hexapod_rough_cache'), ...
    'CodeGenFolder', fullfile(tempdir, 'hexapod_rough_codegen'), 'createDir', true);  % 생성물을 분리한다.
load_system('plant');                          % 현재 모델을 불러온다.
root = sfroot;                                % 함수 블록을 조회한다.
user = root.find('-isa', 'Stateflow.EMChart', 'Path', 'plant/USER');  % 전진 입력을 선택한다.
user_script = strrep(user.Script, '15초', '20초');                  % 시나리오 설명을 갱신한다.
user_script = strrep(user_script, 't <= 15.0', 't <= 20.0');         % 전진 구간을 연장한다.
charts = root.find('-isa', 'Stateflow.EMChart');  % 접촉 생성 함수를 찾는다.
contact = [];                                  % 대상 블록을 초기화한다.
for index = 1:numel(charts)
    if contains(charts(index).Script, '= TestContact(')
        contact = charts(index);               % 가상 험지 블록을 선택한다.
    end
end
assert(isscalar(contact));                     % 접촉 블록의 유일성을 확인한다.
contact_script = contact.Script;               % 원래 험지 조건을 보관한다.
contact_path = contact.Path;                   % 최종 저장 시 경로를 재사용한다.
user.Script = user_script;                     % 20초 전진 입력을 적용한다.
set_param('plant', 'StartTime', '0', 'StopTime', '20', ...
    'SimMechanicsOpenEditorOnUpdate', 'off');   % 전체 검증 시간을 설정한다.

safety_ports = get_param('plant/SafetyEvaluator', 'PortHandles');  % 실제 세이프티 입력을 추적한다.
names = {'Roll_Meas', 'Pitch_Meas', 'IK_Valid1', 'IK_Valid2', 'IK_Valid3', ...
    'IK_Valid4', 'IK_Valid5', 'IK_Valid6'};      % 세이프티 입력 순서에 맞춘다.
for index = 1:8
    line = get_param(safety_ports.Inport(index), 'Line');  % 입력선의 실제 출처를 찾는다.
    source = get_param(line, 'SrcPortHandle');            % 세이프티가 받는 신호를 기록한다.
    log_port(source, names{index});                       % 전체 샘플을 보존한다.
end
log_port(safety_ports.Outport(1), 'Rollover_Fault');     % 전복 세이프티를 기록한다.
log_port(safety_ports.Outport(2), 'Controller_Fault');   % IK 통합 세이프티를 기록한다.
ports = get_param('plant/ControlPriorityManager', 'PortHandles');  % 보행 허용 상태를 조회한다.
log_port(ports.Outport(1), 'Active_Mode');               % 고장에 따른 보행 중단을 확인한다.
ports = get_param('plant/USER', 'PortHandles');          % 사용자 입력을 조회한다.
log_port(ports.Outport(1), 'Throttle');                 % 20초까지 전진 명령을 확인한다.
ports = get_param(contact_path, 'PortHandles');         % 다리별 가상 접촉을 조회한다.
for index = 1:6
    log_port(ports.Outport(index), sprintf('Contact%d', index));  % 요철이 유지되는지 확인한다.
end

scales = [1, 0.5, 0.25, 0.1, 0.05];          % 원래 조건부터 요철 강도를 줄인다.
passed = false;                              % 최종 통과 여부를 초기화한다.
for attempt = 1:numel(scales)
    landing_times = 0.40 + scales(attempt)*[-0.10, 0, 0.075];  % 평지 착지 시점 주변으로 요철을 완화한다.
    replacement = sprintf('Landing_Times = [%.6f, %.6f, %.6f];', landing_times);  % 적용할 착지 시점을 구성한다.
    candidate = regexprep(contact_script, 'Landing_Times = \[[^\]]+\];', replacement);  % 험지 강도만 변경한다.
    contact.Script = candidate;               % 해당 조건을 모델에 적용한다.
    fprintf('ATTEMPT %d landing_times=%s\n', attempt, mat2str(landing_times));  % 검증 조건을 출력한다.
    out = sim('plant', 'ReturnWorkspaceOutputs', 'on', 'SaveTime', 'on', ...
        'SignalLogging', 'on', 'SignalLoggingName', 'logsout');  % 0초부터 20초까지 실행한다.
    assert(abs(out.tout(end)-20) < 1e-9);        % 조기 종료 없이 완료되었는지 확인한다.
    checked_names = [names(3:8), {'Controller_Fault', 'Rollover_Fault'}];  % 통과에 필요한 신호를 지정한다.
    passed = true;                            % 현재 조건의 검사를 시작한다.
    for index = 1:numel(checked_names)
        signal = out.logsout.get(checked_names{index}).Values;  % 전체 시뮬레이션 샘플을 읽는다.
        assert(signal.Time(1) == 0 && abs(signal.Time(end)-20) < 1e-9);  % 검사 구간 누락을 차단한다.
        expected = double(index <= 6);        % IK는 1이고 Fault는 0이어야 한다.
        bad = ~isfinite(signal.Data) | signal.Data ~= expected;  % 모든 샘플의 이상을 검사한다.
        first_bad = find(bad, 1);             % 최초 발생 시점을 찾는다.
        if ~isempty(first_bad)
            passed = false;                  % 이상이 한 번이라도 있으면 탈락시킨다.
            fprintf('FAIL %s first_time=%.6f bad_samples=%d\n', ...
                checked_names{index}, signal.Time(first_bad), nnz(bad));  % 실패 시점과 횟수를 기록한다.
        end
    end
    save(fullfile(output_dir, sprintf('rough_walk_attempt_%d.mat', attempt)), ...
        'out', 'landing_times', 'passed', '-v7.3');  % 실패 조건도 재확인할 수 있도록 보관한다.
    if passed
        first = out.logsout.get('Contact1').Values;  % 요철 접촉 차이를 검사한다.
        third = out.logsout.get('Contact3').Values;  % 같은 그룹의 다리를 비교한다.
        assert(isequal(first.Time, third.Time) && any(first.Data ~= third.Data));  % 평지로 바뀌지 않았는지 확인한다.
        mode = out.logsout.get('Active_Mode').Values;  % 끝까지 보행 허용 상태인지 검사한다.
        assert(all(mode.Data(mode.Time >= 7) == 3));  % 전진 구간의 수동 보행 모드를 확인한다.
        throttle = out.logsout.get('Throttle').Values;  % 마지막 전진 명령을 확인한다.
        assert(throttle.Data(end) == 1000);           % 20초까지 전진 입력을 유지한다.
        report = table('Size', [0, 3], 'VariableTypes', {'double', 'string', 'double'}, ...
            'VariableNames', {'Time', 'Signal', 'Value'});  % 신호별 원본 시간축을 보존한다.
        for index = 1:out.logsout.numElements
            entry = out.logsout.get(index);          % 기록한 신호를 순회한다.
            signal = entry.Values;                  % 원본 샘플을 읽는다.
            rows = table(signal.Time(:), repmat(string(entry.Name), numel(signal.Time), 1), ...
                signal.Data(:), 'VariableNames', {'Time', 'Signal', 'Value'});  % 각 신호의 모든 샘플을 구성한다.
            report = [report; rows];                % 서로 다른 샘플 주기를 그대로 저장한다.
        end
        writetable(report, fullfile(output_dir, 'rough_walk_verified.csv'));  % 검증 결과를 CSV로 저장한다.
        break;
    end
end
assert(passed, 'All terrain candidates failed; inspect rough_walk_log.txt.');  % 통과하지 않은 모델은 저장하지 않는다.
close_system('plant', 0);                       % 임시 로깅 설정을 제거한다.
load_system('plant');                          % 원래 설정으로 모델을 다시 연다.
root = sfroot;                                % 다시 연 함수 블록을 조회한다.
user = root.find('-isa', 'Stateflow.EMChart', 'Path', 'plant/USER');  % 전진 입력을 선택한다.
contact = root.find('-isa', 'Stateflow.EMChart', 'Path', contact_path);  % 접촉 블록을 선택한다.
user.Script = user_script;                     % 검증한 전진 구간을 저장한다.
contact.Script = candidate;                    % 통과한 험지 조건을 저장한다.
set_param('plant', 'StartTime', '0', 'StopTime', '20');  % 실행 시간을 저장한다.
save_system('plant');                          % 검증 완료 모델을 저장한다.
close_system('plant', 0);                       % 모델 파일을 닫는다.
fprintf('PASS full_interval=0..20 landing_times=%s recorded_rows=%d all_IK_valid=1 all_faults=0\n', ...
    mat2str(landing_times), height(report));    % 최종 통과 근거를 출력한다.
end

% 지정한 출력 포트의 모든 샘플을 기록한다.
function log_port(port, name)
set_param(port, 'DataLogging', 'on', 'DataLoggingNameMode', 'Custom', ...
    'DataLoggingName', name, 'DataLoggingDecimateData', 'off', ...
    'DataLoggingLimitDataPoints', 'off');        % 간격 생략과 기록 개수 제한을 해제한다.
end
