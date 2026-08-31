% 기존 시험 입력으로 모델을 실행하고 평가 신호를 저장한다.
function run_evaluation
output_dir = fileparts(mfilename('fullpath'));  % 결과 저장 경로를 설정한다.
model_dir  = fileparts(output_dir);             % 원본 모델 경로를 설정한다.
old_dir    = pwd;                               % 호출 위치를 보관한다.
cleanup    = onCleanup(@() cd(old_dir));         % 종료 시 호출 위치를 복원한다.
cd(model_dir);                                  % 초기화 스크립트의 상대 경로를 유지한다.
diary(fullfile(output_dir, 'simulation_log.txt'));  % 실제 실행 기록을 저장한다.
diary_cleanup = onCleanup(@() diary('off'));        % 오류가 발생해도 기록을 닫는다.

model = 'plant';                              % 기존 모델 이름을 설정한다.
assert(~bdIsLoaded(model), 'Use a separate MATLAB session; plant is already open.');  % 열려 있는 모델의 미저장 작업을 보호한다.
load_system(model);                           % 원본 모델을 메모리에 불러온다.
model_cleanup = onCleanup(@() close_system(model, 0));  % 원본 파일을 저장하지 않고 닫는다.
Simulink.fileGenControl('set', 'CacheFolder', fullfile(output_dir, 'cache'), ...
    'CodeGenFolder', fullfile(output_dir, 'codegen'), 'createDir', true);  % 생성 파일을 분리한다.
properties = get_param(model, 'ObjectParameters');  % 사용 가능한 모델 설정을 확인한다.
if isfield(properties, 'SimMechanicsOpenEditorOnUpdate')
    set_param(model, 'SimMechanicsOpenEditorOnUpdate', 'off');  % 배치 실행 중 뷰어를 열지 않는다.
end

blocks = {'USER', 'DroneController', 'ControlPriorityManager', ...
    'TripodFootTrajectory', 'TripodGaitManager', 'BodyPosturePIOverlay', ...
    'Body2Leg_IK', 'JointRateLimiter', 'Plant', 'FK_Leg2Body', ...
    'BodyPositionEstimator', 'GaitPosePI', 'SafetyEvaluator', 'MATLAB Function'};  % 평가 신호의 출처를 지정한다.
signal_names = strings(0, 1);  % 저장 변수 이름을 수집한다.
signal_paths = strings(0, 1);  % 모델 내 신호 위치를 수집한다.
signal_ports = zeros(0, 1);    % 출력 포트 번호를 수집한다.
for block_index = 1:numel(blocks)
    block_path = [model '/' blocks{block_index}];  % 대상 블록의 전체 경로를 구성한다.
    ports = get_param(block_path, 'PortHandles');  % 출력 포트 핸들을 가져온다.
    outputs = find_system(block_path, 'SearchDepth', 1, 'BlockType', 'Outport');  % 출력 이름을 확인한다.
    for port_index = 1:numel(ports.Outport)
        output_name = sprintf('port%d', port_index);  % 이름이 없는 출력의 대체 이름을 설정한다.
        for output_index = 1:numel(outputs)
            if str2double(get_param(outputs{output_index}, 'Port')) == port_index
                output_name = get_param(outputs{output_index}, 'Name');  % 모델에 정의된 신호 이름을 사용한다.
            end
        end
        name = matlab.lang.makeValidName([blocks{block_index} '__' output_name]);  % CSV 호환 이름을 생성한다.
        set_param(ports.Outport(port_index), 'DataLogging', 'on', ...
            'DataLoggingNameMode', 'Custom', 'DataLoggingName', name);  % 연결된 출력 신호를 기록한다.
        signal_names(end+1, 1) = string(name);        % 저장 이름을 추가한다.
        signal_paths(end+1, 1) = string(block_path);  % 신호 출처를 추가한다.
        signal_ports(end+1, 1) = port_index;         % 출력 포트 번호를 추가한다.
    end
end
signal_map = table(signal_names, signal_paths, signal_ports);  % 추적 가능한 신호 목록을 구성한다.
writetable(signal_map, fullfile(output_dir, 'signal_map.csv'));  % 신호 목록을 저장한다.

fprintf('START %s | MATLAB %s | original input | 0..81 s\n', char(datetime('now')), version);  % 실행 조건을 기록한다.
timer = tic;  % 실제 실행 시간을 측정한다.
out = sim(model, 'StartTime', '0', 'StopTime', '81', ...
    'SignalLogging', 'on', 'SignalLoggingName', 'logsout', ...
    'ReturnWorkspaceOutputs', 'on', 'SaveTime', 'on');  % 기존 시험 입력을 변경하지 않고 실행한다.
elapsed_seconds = toc(timer);  % 실행 소요 시간을 저장한다.
save(fullfile(output_dir, 'simulation_raw.mat'), 'out', 'signal_map', 'elapsed_seconds', '-v7.3');  % 원시 데이터를 보관한다.
fprintf('DONE %.1f wall seconds | %d signals | final time %.6f\n', ...
    elapsed_seconds, out.logsout.numElements, out.tout(end));  % 완료 여부와 기록 개수를 출력한다.
end
