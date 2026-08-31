% 현재 Classical 측정값과 대표 6족 로봇의 문헌 성능을 구분해 비교한다.
function make_figure_08
output_dir = fileparts(mfilename('fullpath'));                   % 결과 저장 경로를 설정한다.
metrics = jsondecode(fileread(fullfile(output_dir, 'metrics.json')));  % 이전 Simulink 실행에서 산출한 값을 불러온다.
names = {'현재 Classical', 'RHex [1]', 'PhantomX AX [2]', 'HAntR [2]'};  % 현재 모델과 문헌 비교 대상을 지정한다.
speed = [NaN 0.55 0.29 0.43];                                   % 실제 주행이 미평가된 현재 모델은 결측값으로 유지한다.
foot = [metrics.foot_rmse_all_legs_mm NaN NaN NaN];              % 동일 정의의 문헌 발끝 오차를 임의로 채우지 않는다.
joint = [metrics.joint_rmse_all_joints_deg NaN NaN NaN];         % 문헌에 없는 관절 RMS 오차를 결측값으로 유지한다.
colors = [0.12 0.36 0.64; 0.26 0.53 0.44; ...
          0.32 0.59 0.68; 0.53 0.55 0.67];                     % 현재 모델과 문헌 로봇을 색상으로 구분한다.

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1650 850]);  % 가로형 보고서 그림을 생성한다.
set(fig, 'DefaultAxesFontName', 'Malgun Gothic', 'DefaultTextFontName', 'Malgun Gothic');  % 한글을 지원하는 글꼴을 지정한다.
label(fig, [0.03 0.905 0.94 0.07], ...
    '사진 8. Classical Controller 및 대표 6족 로봇의 성능 비교', 24, 'bold');  % 변경된 비교 대상을 제목에 표시한다.
label(fig, [0.03 0.848 0.94 0.055], ...
    '현재 Simulink 제어 시험 + 공개 문헌 참고값  |  서로 다른 시험 조건으로 성능 순위 판정 불가', 14, 'normal');  % 비교의 한계를 표시한다.

positions = [0.065 0.32 0.25 0.45; 0.390 0.32 0.25 0.45; 0.715 0.32 0.25 0.45];  % 세 지표의 패널 위치를 설정한다.
values = {speed, foot, joint};                                   % 단위가 다른 지표를 별도 패널로 나눈다.
limits = [0.70 4.6 0.56];                                       % 값과 결측 설명이 보이도록 축 범위를 설정한다.
headings = {'실제 주행 최대 속도', '발끝 RMS 추종 오차', '관절 RMS 추종 오차'};  % 각 패널의 측정 대상을 표시한다.
units = {'속도 (m/s)', '3차원 거리 오차 (mm)', '관절각 오차 (deg)'};           % 지표별 단위를 표시한다.
notes = {'문헌의 실제 로봇 결과', '현재 모델: 6개 다리, 7–10 s', '현재 모델: 18개 관절, 7–10 s'};  % 집계 조건을 구분한다.
for panel = 1:3
    ax = axes(fig, 'Position', positions(panel, :));              % 해당 패널의 좌표축을 생성한다.
    bars = bar(ax, 1:4, values{panel}, 0.58, 'FaceColor', 'flat', 'EdgeColor', 'none');  % 확인된 수치만 막대로 표시한다.
    bars.CData = colors;                                        % 로봇별 색상을 적용한다.
    ylim(ax, [0 limits(panel)]); xlim(ax, [0.4 4.6]);             % 축 범위를 설정한다.
    set(ax, 'XTick', 1:4, 'XTickLabel', names, 'XTickLabelRotation', 20, ...
        'FontSize', 12, 'Box', 'off', 'YGrid', 'on', 'GridAlpha', 0.15);  % 레이블과 격자 가독성을 맞춘다.
    ylabel(ax, units{panel}, 'FontSize', 13);                     % 수치 단위를 표시한다.
    title(ax, headings{panel}, 'FontSize', 18);                   % 패널 제목을 표시한다.
    subtitle(ax, notes{panel}, 'FontSize', 12);                   % 시험 조건을 표시한다.
    for robot = 1:4
        value = values{panel}(robot);                           % 해당 로봇의 지표 값을 가져온다.
        if isnan(value)
            missing = '미확보';                                 % 문헌에서 같은 지표를 찾지 못했음을 표시한다.
            if robot == 1
                missing = '미평가';                             % 현재 모델의 실제 주행 시험이 없음을 표시한다.
            end
            text(ax, robot, limits(panel)*0.20, missing, ...
                'HorizontalAlignment', 'center', 'Color', [0.45 0.47 0.50], 'FontSize', 12);  % 결측값을 0으로 표시하지 않는다.
        else
            text(ax, robot, value+limits(panel)*0.045, sprintf('%.3f', value), ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 15);  % 확인된 수치를 막대 위에 표시한다.
        end
    end
end

label(fig, [0.045 0.19 0.91 0.05], ...
    sprintf('현재 Classical의 내부 위치 추정기 평균 속도: %.4f m/s — 몸체 고정 모델이므로 실제 주행 속도 그래프에서 제외', ...
    metrics.forward_estimated_x_speed_mps), 13, 'normal');        % 추정기 출력을 실제 속도와 분리한다.
label(fig, [0.045 0.135 0.91 0.045], ...
    '미평가 = 해당 실험 없음  /  미확보 = 선택 출처에서 동일 지표 수치 미확보  /  둘 다 0 또는 성능 부족을 의미하지 않음', 12, 'normal');  % 결측 표시의 의미를 설명한다.
label(fig, [0.045 0.085 0.91 0.04], ...
    '[1] Saranli et al., IJRR (2001), Table 1    [2] Cizek et al., IEEE Access (2021), Table 1', 12, 'normal');  % 수치의 문헌 출처를 표시한다.
label(fig, [0.045 0.035 0.91 0.04], ...
    '발끝·관절 오차는 현재 모델의 시뮬레이션 결과이며, 외부 로봇 대비 정확도 우위를 입증한 값은 아니다.', 12, 'normal');  % 검증되지 않은 성능 우위 해석을 방지한다.

exportgraphics(fig, fullfile(output_dir, 'figure_08_hexapod_comparison.png'), 'Resolution', 220);  % 보고서용 고해상도 이미지를 저장한다.
savefig(fig, fullfile(output_dir, 'figure_08_hexapod_comparison.fig'));  % 편집 가능한 MATLAB 그림을 저장한다.
close(fig);                                                     % 그림 리소스를 반환한다.

Robot = string(names)';                                         % CSV용 로봇 이름을 구성한다.
ActualMaxSpeed_mps = speed';                                    % 실제 속도의 문헌 수치를 저장한다.
FootRMS_mm = foot';                                             % 발끝 RMS 오차와 결측값을 보존한다.
JointRMS_deg = joint';                                          % 관절 RMS 오차와 결측값을 보존한다.
Evidence = ["Simulation: body fixed, 7 <= t < 10 s"; ...
    "RHex (2001), Table 1"; "Cizek et al. (2021), Table 1"; "Cizek et al. (2021), Table 1"];  % 데이터 출처를 붙인다.
Source = ["metrics.json"; ...
    "https://www.ri.cmu.edu/pub_files/pub4/saranli_uluc_2001_1/saranli_uluc_2001_1.pdf"; ...
    "https://comrob.fel.cvut.cz/papers/access21hantr.pdf"; ...
    "https://comrob.fel.cvut.cz/papers/access21hantr.pdf"];         % 각 수치의 근거 링크를 저장한다.
comparison = table(Robot, ActualMaxSpeed_mps, FootRMS_mm, JointRMS_deg, Evidence, Source);  % 비교 데이터를 구성한다.
writetable(comparison, fullfile(output_dir, 'hexapod_comparison.csv'), 'Encoding', 'UTF-8');  % 출처를 포함한 CSV를 저장한다.
end

% 그림의 제목과 주석을 정규화 좌표에 배치한다.
function label(fig, position, content, font_size, weight)
annotation(fig, 'textbox', position, 'String', content, ...
    'Interpreter', 'none', 'FontName', 'Malgun Gothic', ...
    'FontSize', font_size, 'FontWeight', weight, 'Color', [0.18 0.22 0.28], ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'EdgeColor', 'none', 'Margin', 0);                            % 한글과 출처 설명을 표시한다.
end
