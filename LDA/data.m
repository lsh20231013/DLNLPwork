% 数据统计处理和分析

%% 1. 准备数据
% 主题数
topics = [5 10 20 50];

% 不同段落长度下的分类准确率
accuracy_100 = [0.356588, 0.440528, 0.535970, 0.613763];
accuracy_500 = [0.483668, 0.572416, 0.689424, 0.791391];
accuracy_1000 = [0.563374, 0.720562, 0.762578, 0.836415];
accuracy_3000 = [0.533009, 0.723357, 0.836188, 0.834604];

% 运行时间数据（转换为分钟）
runtime_100 = [51+11/60, 52+35/60, 52+36/60, 54+23/60];
runtime_500 = [14+13/60, 12+48/60, 14+41/60, 42+02/60];
runtime_1000 = [7+57/60, 7+24/60, 8+05/60, 35+01/60];
runtime_3000 = [3+43/60, 5+35/60, 11+20/60, 15+19/60];

% 段落长度
para_lengths = [100, 500, 1000, 3000];

%% 2. 绘制准确率随主题数变化曲线
figure;
hold on;
plot(topics, accuracy_100, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '100 词');
plot(topics, accuracy_500, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '500 词');
plot(topics, accuracy_1000, '-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '1000 词');
plot(topics, accuracy_3000, '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '3000 词');

% 设置图像格式
xlabel('LDA 主题数');
ylabel('分类准确率');
title('不同主题数下的分类准确率对比');
legend('Location', 'southeast');
grid on;
hold off;

%% 3. 绘制运行时间随主题数变化曲线
figure;
hold on;
plot(topics, runtime_100, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '100 词');
plot(topics, runtime_500, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '500 词');
plot(topics, runtime_1000, '-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '1000 词');
plot(topics, runtime_3000, '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '3000 词');

% 设置图像格式
xlabel('LDA 主题数');
ylabel('运行时间 (分钟)');
title('不同主题数下的运行时间对比');
legend('Location', 'northeast');
grid on;
hold off;

%% 4. 绘制准确率随段落长度变化曲线
figure;
hold on;
plot(para_lengths, [accuracy_100(1), accuracy_500(1), accuracy_1000(1), accuracy_3000(1)], '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '5 主题');
plot(para_lengths, [accuracy_100(2), accuracy_500(2), accuracy_1000(2), accuracy_3000(2)], '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '10 主题');
plot(para_lengths, [accuracy_100(3), accuracy_500(3), accuracy_1000(3), accuracy_3000(3)], '-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '20 主题');
plot(para_lengths, [accuracy_100(4), accuracy_500(4), accuracy_1000(4), accuracy_3000(4)], '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '50 主题');

% 设置图像格式
xlabel('段落长度 (词)');
ylabel('分类准确率');
title('不同段落长度下的分类准确率对比');
legend('Location', 'southeast');
set(gca, 'XScale', 'log'); % 对数坐标更清晰显示数据
grid on;
hold off;

%% 5. 绘制运行时间随段落长度变化曲线
figure;
hold on;
plot(para_lengths, [runtime_100(1), runtime_500(1), runtime_1000(1), runtime_3000(1)], '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '5 主题');
plot(para_lengths, [runtime_100(2), runtime_500(2), runtime_1000(2), runtime_3000(2)], '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '10 主题');
plot(para_lengths, [runtime_100(3), runtime_500(3), runtime_1000(3), runtime_3000(3)], '-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '20 主题');
plot(para_lengths, [runtime_100(4), runtime_500(4), runtime_1000(4), runtime_3000(4)], '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '50 主题');

% 设置图像格式
xlabel('段落长度 (词)');
ylabel('运行时间 (分钟)');
title('不同段落长度下的运行时间对比');
legend('Location', 'northeast');
set(gca, 'XScale', 'log'); % 对数坐标更清晰显示数据
grid on;
hold off;

%% 6. 创建3D曲面图展示准确率与段落长度和主题数的关系
[TOPICS, LENGTHS] = meshgrid(topics, para_lengths);

% 准备准确率矩阵
ACCURACY = [accuracy_100; accuracy_500; accuracy_1000; accuracy_3000];

figure;
surf(TOPICS, LENGTHS, ACCURACY);
xlabel('主题数');
ylabel('段落长度 (词)');
zlabel('分类准确率');
title('分类准确率与段落长度和主题数的关系');
colorbar;
grid on;

%% 7. 创建3D曲面图展示运行时间与段落长度和主题数的关系
RUNTIME = [runtime_100; runtime_500; runtime_1000; runtime_3000];

figure;
surf(TOPICS, LENGTHS, RUNTIME);
xlabel('主题数');
ylabel('段落长度 (词)');
zlabel('运行时间 (分钟)');
title('运行时间与段落长度和主题数的关系');
colorbar;
grid on;

%% 8. 计算并显示最高准确率配置
[max_acc, max_idx] = max(ACCURACY(:));
[row, col] = ind2sub(size(ACCURACY), max_idx);
fprintf('最高准确率配置:\n');
fprintf('段落长度: %d 词\n', para_lengths(row));
fprintf('主题数: %d\n', topics(col));
fprintf('准确率: %.4f\n', max_acc);
fprintf('运行时间: %.2f 分钟\n\n', RUNTIME(row, col));

%% 9. 计算并显示最佳性价比配置（准确率/时间比）
acc_time_ratio = ACCURACY ./ RUNTIME;
[max_ratio, max_idx] = max(acc_time_ratio(:));
[row, col] = ind2sub(size(acc_time_ratio), max_idx);
fprintf('最佳性价比配置(准确率/时间):\n');
fprintf('段落长度: %d 词\n', para_lengths(row));
fprintf('主题数: %d\n', topics(col));
fprintf('准确率: %.4f\n', ACCURACY(row, col));
fprintf('运行时间: %.2f 分钟\n', RUNTIME(row, col));
fprintf('准确率/时间比: %.4f\n', max_ratio);

%% 10. 绘制字和词在不同主题数下的准确率曲线
wordaccuracy_100 = [0.184152, 0.263938, 0.190494, 0.244035];
wordaccuracy_3000 = [ 0.4918, 0.1431, 0.3222, 0.4389];
figure;
hold on;
plot(topics, accuracy_100, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '100 词');
plot(topics, wordaccuracy_100, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '100 字');
plot(topics, accuracy_3000, '-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '3000 词');
plot(topics, wordaccuracy_3000, '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '3000 字');

% 设置图像格式
xlabel('LDA 主题数');
ylabel('分类准确率');
title('字和词在不同主题数下的准确率对比');
legend('Location', 'northeast');
grid on;
hold off;

%% 11. 绘制字和词在不同主题数下的运行时间曲线
wordruntime_100 = [60+26+45/60, 60+32+41/60, 60+25+6/60, 60+52+10/60];
wordruntime_3000 = [5+33/60, 1+16/60, 22+53/60, 26+40/60];
figure;
hold on;
plot(topics, runtime_100, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '100 词');
plot(topics, wordruntime_100, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '100 字');
plot(topics, runtime_3000, '-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '3000 词');
plot(topics, wordruntime_3000, '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '3000 字');

% 设置图像格式
xlabel('LDA 主题数');
ylabel('运行时间 (分钟)');
title('字和词在不同主题数下的运行时间对比');
legend('Location', 'northeast');
grid on;
hold off;
