% 저장된 가상 자세와 문헌 실측 참고값을 출처별로 구분해 표시한다.
function make_body_stability_figure
output_dir = fileparts(mfilename('fullpath'));                        % 결과 경로를 설정한다.
saved = load(fullfile(output_dir, 'simulation_raw.mat'), 'out');      % 이전 실행의 원본 신호를 불러온다.
t = (0:0.005:81)';                                                   % 기존 후처리와 같은 시간축을 설정한다.
roll = read_signal(saved.out.logsout, 'Plant__Roll_Meas', t)*180/pi;   % 가상 Roll을 도 단위로 변환한다.
pitch = read_signal(saved.out.logsout, 'Plant__Pitch_Meas', t)*180/pi; % 가상 Pitch를 도 단위로 변환한다.
roll_ref = read_signal(saved.out.logsout, 'DroneController__port15', t)*180/pi;   % Roll 기준을 도 단위로 변환한다.
pitch_ref = read_signal(saved.out.logsout, 'DroneController__port16', t)*180/pi; % Pitch 기준을 도 단위로 변환한다.
forward = t >= 7 & t < 10;                                           % 기존 전진 명령 구간을 유지한다.
assert(all(abs([roll_ref(forward); pitch_ref(forward)]) < 1e-10), ...
    'Forward posture reference must be zero.');                      % 수평 기준의 평가 구간인지 확인한다.
attitude = [roll pitch];                                             % 두 축의 가상 자세를 묶는다.
reference = [roll_ref pitch_ref];                                    % 같은 축 순서로 자세 기준을 묶는다.
error_deg = attitude-reference;                                     % 목표 자세에 대한 오차를 계산한다.
virtual_rmse = sqrt(mean(error_deg(forward, :).^2, 1));              % 전진 구간의 가상 자세 RMSE를 계산한다.
literature_rmse = [0.390 0.366];                                     % HexWalker II의 Table 5 Tripod 실측값을 입력한다.
values = [virtual_rmse; literature_rmse];                            % 출처가 다른 값을 별도 행으로 유지한다.

metrics = struct;                                                   % 가상 데이터 전용 요약을 생성한다.
metrics.data_kind = 'Command-derived VirtualIMU, not physical body motion';  % 가상 신호의 성격을 명시한다.
metrics.source = 'simulation_raw.mat, original plant.slx, 0-81 s';   % 재실행이 아닌 원본 후처리임을 기록한다.
metrics.window_s = [7 10];                                          % 끝 시각을 제외한 평가 구간을 기록한다.
metrics.sample_count = nnz(forward);                                % 실제 집계 표본 수를 기록한다.
metrics.sample_interval_s = 0.005;                                  % 공통 시간 간격을 기록한다.
metrics.virtual_roll_rmse_deg = virtual_rmse(1);                     % 가상 Roll RMSE를 저장한다.
metrics.virtual_pitch_rmse_deg = virtual_rmse(2);                    % 가상 Pitch RMSE를 저장한다.
metrics.virtual_forward_max_abs_deg = max(abs(attitude(forward, :)), [], 1);  % 전진 중 최대 절댓값을 저장한다.
metrics.virtual_forward_peak_to_peak_deg = ...
    max(attitude(forward, :), [], 1)-min(attitude(forward, :), [], 1); % 전진 중 최대–최소 변화폭을 저장한다.
metrics.virtual_full_run_max_abs_deg = max(abs(attitude), [], 1);    % 의도적 자세 변경을 포함한 전체 최대값을 저장한다.
metrics.virtual_full_run_peak_to_peak_deg = ...
    max(attitude, [], 1)-min(attitude, [], 1);                        % 전체 시험의 가상 자세 변화폭을 저장한다.
metrics.limit = 'Not a walking stability validation; no ranking or improvement claim.';  % 해석 범위를 제한한다.
fid = fopen(fullfile(output_dir, 'virtual_imu_metrics.json'), 'w', 'n', 'UTF-8');  % 가상 수치를 별도 파일로 저장한다.
assert(fid >= 0, 'Cannot write metrics.');                           % 저장 파일을 열었는지 확인한다.
cleanup = onCleanup(@() fclose(fid));                               % 오류가 발생해도 파일을 닫는다.
fprintf(fid, '%s\n', jsonencode(metrics, 'PrettyPrint', true));       % 수치와 출처를 함께 기록한다.
clear cleanup;                                                      % 기록을 완료하고 파일을 닫는다.

Robot = ["Our Classical - VirtualIMU"; "HexWalker II - Tripod"];     % 가상값과 실측값을 이름에서도 구분한다.
RollRMSE_deg = values(:, 1);                                        % Roll 비교 열을 구성한다.
PitchRMSE_deg = values(:, 2);                                       % Pitch 비교 열을 구성한다.
Evidence = ["Command-derived virtual signal; not physical stability"; ...
    "Physical robot experiment; Zhang et al. 2021, Table 5"];       % 각 행의 증거 유형을 표시한다.
Source = ["simulation_raw.mat; 7 <= t < 10 s"; ...
    "https://www.mdpi.com/2076-3417/11/8/3714"];                    % 원본 데이터와 문헌을 연결한다.
writetable(table(Robot, RollRMSE_deg, PitchRMSE_deg, Evidence, Source), ...
    fullfile(output_dir, 'body_attitude_comparison.csv'), 'Encoding', 'UTF-8');  % 출처를 포함한 비교표를 저장한다.
trace = table(t(forward), roll(forward), pitch(forward), roll_ref(forward), ...
    pitch_ref(forward), 'VariableNames', {'Time_s', 'VirtualRoll_deg', ...
    'VirtualPitch_deg', 'RollReference_deg', 'PitchReference_deg'});  % 그림에 사용한 표본을 별도로 정리한다.
writetable(trace, fullfile(output_dir, 'virtual_imu_forward.csv'));  % 계산에 사용한 신호를 저장한다.

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1500 850]);  % 보고서용 가로 그림을 생성한다.
set(fig, 'DefaultAxesFontName', 'Malgun Gothic', 'DefaultTextFontName', 'Malgun Gothic');  % 한글 글꼴을 지정한다.
label(fig, [0.04 0.90 0.92 0.07], ...
    '사진 8. Classical Controller 가상 자세와 기존 6족 로봇의 참고값', 23, 'bold');  % 그림의 비교 범위를 제목에 표시한다.
label(fig, [0.04 0.83 0.92 0.06], ...
    '명령 기반 VirtualIMU와 문헌 실측값  |  실제 보행 안정성 검증·성능 순위 비교 아님', 15, 'bold');  % 핵심 제한을 눈에 띄게 표시한다.
positions = [0.09 0.31 0.35 0.43; 0.59 0.31 0.35 0.43];           % 두 축의 패널을 배치한다.
names = {'우리 Classical (가상)', 'HexWalker II (실측)'};             % 눈금에서도 데이터 출처를 구분한다.
headings = {'Roll RMSE', 'Pitch RMSE'};                              % 두 자세 축의 제목을 설정한다.
colors = [0.88 0.48 0.16; 0.14 0.39 0.64];                         % 가상값과 실측값의 색을 구분한다.
for panel = 1:2
    ax = axes(fig, 'Position', positions(panel, :));                 % 해당 자세 축의 패널을 생성한다.
    bars = bar(ax, 1:2, values(:, panel), 0.48, 'FaceColor', 'flat', 'EdgeColor', 'none');  % 두 출처의 수치를 표시한다.
    bars.CData = colors;                                           % 출처별 색을 적용한다.
    ylim(ax, [0 max(0.5, max(values(:, panel))*1.25)]);              % 0과 수치 레이블이 보이는 축 범위를 설정한다.
    xlim(ax, [0.4 2.6]);                                           % 막대 좌우 여백을 확보한다.
    set(ax, 'XTick', 1:2, 'XTickLabel', names, 'FontSize', 14, ...
        'YGrid', 'on', 'GridAlpha', 0.15, 'Box', 'off');             % 비교 레이블과 격자를 표시한다.
    ylabel(ax, '기준 자세 대비 RMSE (°)', 'FontSize', 14);           % 오차 정의와 단위를 표시한다.
    title(ax, headings{panel}, 'FontSize', 20);                      % 자세 축의 제목을 표시한다.
    limits = ylim(ax);                                             % 레이블 간격을 축 범위에 맞춘다.
    for row = 1:2
        text(ax, row, values(row, panel)+limits(2)*0.055, ...
            sprintf('%.3f°', values(row, panel)), ...
            'HorizontalAlignment', 'center', 'FontSize', 20, 'FontWeight', 'bold');  % 0을 포함해 계산값을 명시한다.
    end
end
label(fig, [0.04 0.195 0.92 0.05], ...
    '우리 값: 기존 Simulink 실행의 전진 명령 구간 7 ≤ t < 10 s, 목표 Roll·Pitch = 0°', 14, 'normal');  % 평가 구간을 표시한다.
label(fig, [0.04 0.14 0.92 0.05], ...
    '가상 출력이 0°여도 몸체가 흔들리지 않는다는 뜻은 아님: 몸체 고정·명령 기반 자세·가상 접촉 모델', 14, 'normal');  % 0 오차의 오해를 방지한다.
label(fig, [0.04 0.085 0.92 0.05], ...
    'HexWalker II: Tripod 실험, 12 s / 이동 거리 0.345 m  |  시험 조건과 데이터 생성 방식이 다름', 13, 'normal');  % 문헌의 조건 차이를 설명한다.
label(fig, [0.04 0.03 0.92 0.05], ...
    '출처: Zhang et al., Applied Sciences 11(8), 3714 (2021), Table 5  ·  DOI: 10.3390/app11083714', 12, 'normal');  % 문헌을 그림에 연결한다.
exportgraphics(fig, fullfile(output_dir, 'figure_08_body_attitude_comparison.png'), 'Resolution', 200);  % 고해상도 이미지를 저장한다.
savefig(fig, fullfile(output_dir, 'figure_08_body_attitude_comparison.fig'));  % 편집 가능한 그림을 저장한다.
close(fig);                                                        % 그림 리소스를 반환한다.
disp(metrics);                                                     % 계산 수치를 실행 로그에도 표시한다.
end

% 원본 가상 신호를 공통 시간축으로 보간한다.
function values = read_signal(logs, name, t)
signal = logs.getElement(name).Values;                              % 이름으로 원본 신호를 가져온다.
raw = double(reshape(signal.Data, [], 1));                           % 스칼라 신호를 열벡터로 정리한다.
[times, indices] = unique(signal.Time, 'last');                      % 같은 시각의 최종 표본을 선택한다.
assert(numel(raw) == numel(signal.Time), 'Scalar signal required.'); % 신호 차원을 확인한다.
assert(times(1) <= t(1) && times(end) >= t(end), 'Incomplete signal.');  % 전체 구간이 기록됐는지 확인한다.
values = interp1(times, raw(indices), t, 'linear');                  % 기존 평가와 같은 선형 보간을 적용한다.
assert(all(isfinite(values)), 'Nonfinite signal.');                 % 잘못된 값의 집계를 차단한다.
end

% 제목과 해석 제한을 그림의 고정 위치에 표시한다.
function label(fig, position, content, font_size, weight)
annotation(fig, 'textbox', position, 'String', content, 'Interpreter', 'none', ...
    'FontName', 'Malgun Gothic', 'FontSize', font_size, 'FontWeight', weight, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'EdgeColor', 'none', 'Margin', 0);                              % 한글 설명을 패널 밖에 배치한다.
end
