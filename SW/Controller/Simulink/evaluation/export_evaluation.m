% 저장된 실행 데이터에서 그래프와 수치 요약을 생성한다.
function export_evaluation
output_dir = fileparts(mfilename('fullpath'));  % 결과 경로를 설정한다.
saved = load(fullfile(output_dir, 'simulation_raw.mat'));  % 실제 시뮬레이션 결과를 불러온다.
logs = saved.out.logsout;                       % 기록된 신호를 가져온다.
t = (0:0.005:81)';                              % 제어 주기의 공통 시간축을 생성한다.
data = table(t, 'VariableNames', {'Time_s'});    % CSV 출력 표를 생성한다.
for index = 1:logs.numElements
    signal = logs.getElement(index);                     % 원본 기록을 가져온다.
    values = double(reshape(signal.Values.Data, [], 1));  % 스칼라 신호를 열벡터로 정리한다.
    [times, unique_index] = unique(signal.Values.Time, 'last');  % 같은 시각의 최종 값을 선택한다.
    assert(numel(values) == numel(signal.Values.Time), 'Scalar signals required.');  % 신호 차원을 검증한다.
    assert(times(1) <= t(1) && times(end) >= t(end), 'Incomplete signal.');  % 전체 구간의 기록을 확인한다.
    data.(signal.Name) = interp1(times, values(unique_index), t, 'linear');  % 공통 시간축에 값을 보간한다.
end
assert(all(isfinite(data{:, :}), 'all'), 'Nonfinite simulation data.');  % 유효하지 않은 결과를 차단한다.
writetable(data, fullfile(output_dir, 'signals_200Hz.csv'));  % 모든 기록 신호를 CSV로 저장한다.

reference = block_values(data, saved.signal_map, 'BodyPosturePIOverlay', 1:18);  % 최종 발끝 목표를 가져온다.
response  = block_values(data, saved.signal_map, 'FK_Leg2Body', 1:18);            % 관절 응답의 정기구학 결과를 가져온다.
q_target  = block_values(data, saved.signal_map, 'JointRateLimiter', 1:18);     % 속도 제한 후 관절 목표를 가져온다.
q_actual  = block_values(data, saved.signal_map, 'Plant', 1:18);                % 관절 동역학 응답을 가져온다.
body_est  = block_values(data, saved.signal_map, 'BodyPositionEstimator', 1:2);  % 몸체 위치 추정값을 가져온다.
body_ref  = block_values(data, saved.signal_map, 'GaitPosePI', 5:6);             % 내부 위치 기준을 가져온다.
imu       = block_values(data, saved.signal_map, 'Plant', 25:27);               % 가상 IMU 출력을 가져온다.
pose_ref  = block_values(data, saved.signal_map, 'DroneController', 15:16);      % 자세 기준을 가져온다.
faults    = block_values(data, saved.signal_map, 'SafetyEvaluator', 1:2);        % Fault 출력 상태를 가져온다.
ik_valid  = block_values(data, saved.signal_map, 'Body2Leg_IK', 19:24);          % 역기구학 유효 신호를 가져온다.
forward   = t >= 7 & t < 10;                                                  % 원래 전진 명령 구간을 선택한다.
cycle     = t >= 8 & t <= 9;                                                  % 전진 중 1초 궤적을 선택한다.
tracking_error = reshape(response-reference, [], 3, 6);                       % 다리별 좌표 오차를 구성한다.
foot_error = reshape(sqrt(sum(tracking_error.^2, 2)), [], 6);                  % 3차원 발끝 거리 오차를 계산한다.
joint_error = (q_actual-q_target)*180/pi;                                      % 관절 추종 오차를 도 단위로 변환한다.
path_error = sqrt(sum((body_est-body_ref).^2, 2));                             % 내부 기준 대비 추정 위치 차이를 계산한다.

leg = (1:6)';                                                          % 다리 번호를 생성한다.
rmse_mm = sqrt(mean(foot_error(forward, :).^2, 1))'*1000;                 % 전진 구간의 발끝 RMS 거리 오차를 계산한다.
max_mm = max(foot_error(forward, :), [], 1)'*1000;                        % 전진 구간의 최대 거리 오차를 계산한다.
excursion_cm = (max(response(forward, 3:3:18), [], 1)- ...
    min(response(forward, 3:3:18), [], 1))'*100;                         % 몸체 좌표계에서 발끝 높이 변화폭을 계산한다.
leg_metrics = table(leg, rmse_mm, max_mm, excursion_cm);                 % 다리별 수치를 정리한다.
writetable(leg_metrics, fullfile(output_dir, 'foot_metrics.csv'));       % 다리별 수치를 저장한다.
joint_rmse = sqrt(mean(joint_error(forward, :).^2, 1));                  % 전진 구간의 관절 RMS 오차를 계산한다.
joint_metrics = table((1:18)', joint_rmse', ...
    'VariableNames', {'JointIndex', 'RMSE_deg'});                        % 18개 관절 평가 결과를 구성한다.
writetable(joint_metrics, fullfile(output_dir, 'joint_metrics.csv'));    % 관절별 수치를 저장한다.

metrics = struct;                                                              % 측정 조건을 포함할 요약을 생성한다.
metrics.simulation_stop_s = saved.out.tout(end);                                 % 실제 종료 시각을 기록한다.
metrics.wall_time_s = saved.elapsed_seconds;                                    % 실제 실행 시간을 기록한다.
metrics.logged_signals = logs.numElements;                                     % 원시 기록 신호 수를 기록한다.
metrics.forward_window_s = [7 10];                                              % 오차 평가 구간을 기록한다.
metrics.foot_rmse_all_legs_mm = sqrt(mean(foot_error(forward, :).^2, 'all'))*1000;  % 전체 다리의 발끝 RMS 오차를 계산한다.
metrics.joint_rmse_all_joints_deg = sqrt(mean(joint_error(forward, :).^2, 'all'));  % 전체 관절의 RMS 오차를 계산한다.
metrics.max_abs_virtual_roll_deg = max(abs(imu(:, 1)))*180/pi;                    % 가상 Roll의 최대 절댓값을 계산한다.
metrics.max_abs_virtual_pitch_deg = max(abs(imu(:, 2)))*180/pi;                   % 가상 Pitch의 최대 절댓값을 계산한다.
metrics.virtual_roll_peak_to_peak_deg = (max(imu(:, 1))-min(imu(:, 1)))*180/pi;   % 가상 Roll의 전체 변화폭을 계산한다.
metrics.virtual_pitch_peak_to_peak_deg = (max(imu(:, 2))-min(imu(:, 2)))*180/pi;  % 가상 Pitch의 전체 변화폭을 계산한다.
forward_indices = find(forward);                                               % 전진 구간의 양 끝을 찾는다.
metrics.forward_estimated_x_speed_mps = ...
    (body_est(forward_indices(end), 1)-body_est(forward_indices(1), 1))/ ...
    (t(forward_indices(end))-t(forward_indices(1)));                              % 위치 추정기의 전진 평균 속도를 계산한다.
metrics.mean_internal_xy_error_forward_m = mean(path_error(forward));           % 전진 구간 내부 위치 차이의 평균을 계산한다.
metrics.max_rollover_fault = max(faults(:, 1));                                  % 가상 전복 Fault의 발생 여부를 기록한다.
metrics.max_controller_fault = max(faults(:, 2));                                % 제어기 Fault의 발생 여부를 기록한다.
metrics.all_legs_ik_valid_sample_pct = mean(all(ik_valid > 0.5, 2))*100;          % 여섯 다리 모두 유효한 표본 비율을 계산한다.
metrics.scope = 'Body fixed to world; synthetic contact and VirtualIMU; no terrain or Residual-RL validation.';  % 해석 한계를 명시한다.
fid = fopen(fullfile(output_dir, 'metrics.json'), 'w', 'n', 'UTF-8');             % UTF-8 요약 파일을 생성한다.
fprintf(fid, '%s\n', jsonencode(metrics, 'PrettyPrint', true));                  % 기계 판독 가능한 수치를 저장한다.
fclose(fid);                                                                    % 요약 파일을 닫는다.
disp(metrics);                                                                  % 결과를 실행 로그에도 출력한다.

set(groot, 'defaultAxesFontName', 'Arial', 'defaultAxesFontSize', 11, ...
    'defaultLineLineWidth', 1.6, 'defaultFigureColor', 'w');  % 보고서용 그래프 스타일을 설정한다.
colors = [0.12 0.35 0.64; 0.88 0.36 0.13];                  % 목표와 응답의 색상을 구분한다.
fig = figure('Visible', 'off', 'Position', [100 100 1400 900]);  % 고해상도 그림을 생성한다.
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');  % 네 개 패널을 배치한다.
title(layout, 'MATLAB/Simulink foot trajectory and tracking', 'FontSize', 20);   % 그림 제목을 작성한다.
subtitle(layout, 'Original plant.slx | Body-fixed model | Forward command: 7-10 s', 'FontSize', 12);  % 시험 범위를 표시한다.
nexttile;  % 대표 다리의 입체 궤적 패널을 선택한다.
plot3(reference(cycle, 1)*1000, reference(cycle, 2)*1000, reference(cycle, 3)*1000, ...
    '--', 'Color', colors(1, :));  % 1번 다리의 목표 궤적을 표시한다.
hold on;  % 응답 궤적을 겹쳐 그린다.
plot3(response(cycle, 1)*1000, response(cycle, 2)*1000, response(cycle, 3)*1000, ...
    'Color', colors(2, :));        % 1번 다리의 정기구학 응답을 표시한다.
grid on; axis equal; view(38, 25); yticks([-340 -290]);  % 좌표 비율을 유지하고 짧은 축의 눈금 겹침을 줄인다.
xlabel('Body X (mm)'); ylabel('Body Y (mm)'); zlabel('Body Z (mm)');  % 좌표계와 단위를 표시한다.
title('Leg 1: 3D trajectory, 8-9 s'); legend('Target', 'Joint-response FK', 'Location', 'best');  % 신호 출처를 구분한다.
nexttile;  % 발끝 높이 응답 패널을 선택한다.
plot(t(forward), reference(forward, 3)*1000, '--', 'Color', colors(1, :)); hold on;  % 목표 높이를 표시한다.
plot(t(forward), response(forward, 3)*1000, 'Color', colors(2, :));                % 응답 높이를 표시한다.
grid on; xlabel('Time (s)'); ylabel('Body Z (mm)'); title('Leg 1: vertical tracking');  % 패널 정보를 작성한다.
legend('Target', 'Joint-response FK', 'Location', 'best');  % 목표와 응답 범례를 표시한다.
nexttile;  % 거리 오차 패널을 선택한다.
plot(t(forward), foot_error(forward, :)*1000); grid on;  % 여섯 다리의 거리 오차를 표시한다.
xlabel('Time (s)'); ylabel('3D tracking error (mm)'); title('All legs: target-to-response error');  % 오차 정의를 표시한다.
legend(compose('Leg %d', 1:6), 'Location', 'best', 'NumColumns', 3);  % 다리 번호를 구분한다.
nexttile;  % RMS 요약 패널을 선택한다.
bar(leg, rmse_mm, 0.65, 'FaceColor', colors(1, :)); grid on;  % 다리별 RMS 오차를 비교한다.
xlabel('Leg'); ylabel('3D RMS tracking error (mm)'); title('Forward segment: 7 <= t < 10 s');  % 집계 구간을 명시한다.
xticks(1:6);  % 여섯 다리 눈금을 표시한다.
text(leg, rmse_mm+0.08, compose('%.2f', rmse_mm), 'HorizontalAlignment', 'center');  % 막대 위에 RMS 수치를 표시한다.
exportgraphics(fig, fullfile(output_dir, 'figure_07_foot_trajectory.png'), 'Resolution', 220);  % 보고서용 PNG를 저장한다.
savefig(fig, fullfile(output_dir, 'figure_07_foot_trajectory.fig'));  % 편집 가능한 MATLAB 그림을 저장한다.
close(fig);  % 그림 리소스를 반환한다.

fig = figure('Visible', 'off', 'Position', [100 100 1400 900]);  % 제어 응답 그림을 생성한다.
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');  % 네 개 패널을 배치한다.
title(layout, 'Classical controller: existing-model results', 'FontSize', 20);  % 비교 실험이 아님을 제목에 표시한다.
subtitle(layout, 'Single 81 s run | Virtual attitude is command-derived | No Residual-RL comparison', 'FontSize', 12);  % 데이터 한계를 표시한다.
nexttile;  % Roll 패널을 선택한다.
plot(t, pose_ref(:, 1)*180/pi, '--', 'Color', colors(1, :)); hold on;  % Roll 기준을 표시한다.
plot(t, imu(:, 1)*180/pi, 'Color', colors(2, :)); grid on;  % 가상 Roll 응답을 표시한다.
xlabel('Time (s)'); ylabel('Roll (deg)'); title('Virtual Roll: posture command test');  % 자세 시험임을 표시한다.
legend('Reference', 'VirtualIMU', 'Location', 'best');  % 가상 센서 신호를 명시한다.
nexttile;  % Pitch 패널을 선택한다.
plot(t, pose_ref(:, 2)*180/pi, '--', 'Color', colors(1, :)); hold on;  % Pitch 기준을 표시한다.
plot(t, imu(:, 2)*180/pi, 'Color', colors(2, :)); grid on;  % 가상 Pitch 응답을 표시한다.
xlabel('Time (s)'); ylabel('Pitch (deg)'); title('Virtual Pitch: posture command test');  % 자세 시험임을 표시한다.
legend('Reference', 'VirtualIMU', 'Location', 'best');  % 신호 출처를 구분한다.
nexttile;  % 관절 추종 패널을 선택한다.
plot(t(forward), q_target(forward, 2)*180/pi, '--', 'Color', colors(1, :)); hold on;  % 대표 관절 목표를 표시한다.
plot(t(forward), q_actual(forward, 2)*180/pi, 'Color', colors(2, :)); grid on;  % 대표 관절 응답을 표시한다.
xlabel('Time (s)'); ylabel('Joint angle (deg)'); title('Leg 1 / Joint 2: dynamic response');  % 관절과 단위를 표시한다.
legend('Rate-limited target', 'Plant response', 'Location', 'best');  % 목표 처리 단계를 명시한다.
nexttile;  % 전체 관절 RMS 패널을 선택한다.
bar(reshape(joint_rmse, 3, 6)'); grid on;  % 다리별 세 관절의 RMS 오차를 표시한다.
xlabel('Leg'); ylabel('Joint RMS error (deg)'); title('18 joints: forward segment, 7-10 s');  % 평가 구간을 명시한다.
legend('Joint 1', 'Joint 2', 'Joint 3', 'Location', 'northoutside', 'Orientation', 'horizontal');  % 막대를 가리지 않게 범례를 표시한다.
exportgraphics(fig, fullfile(output_dir, 'classical_control_results.png'), 'Resolution', 220);  % 보조 그래프를 저장한다.
savefig(fig, fullfile(output_dir, 'classical_control_results.fig'));  % 재편집용 그림을 저장한다.
close(fig);  % 그림 리소스를 반환한다.
end

% 블록 이름과 출력 포트로 공통 시간축의 신호를 선택한다.
function values = block_values(data, signal_map, block_name, ports)
values = zeros(height(data), numel(ports));  % 출력 행렬을 준비한다.
for index = 1:numel(ports)
    row = signal_map.signal_paths == "plant/"+string(block_name) & ...
        signal_map.signal_ports == ports(index);  % 원본 모델의 출력 포트를 찾는다.
    assert(nnz(row) == 1, 'Ambiguous signal map.');  % 출처가 유일한지 확인한다.
    name = char(signal_map.signal_names(row));     % 저장된 변수 이름을 가져온다.
    values(:, index) = data.(name);               % 요청한 신호를 결과에 배치한다.
end
end
