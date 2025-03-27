% ����ͳ�ƴ���ͷ���

%% 1. ׼������
% ������
topics = [5 10 20 50];

% ��ͬ���䳤���µķ���׼ȷ��
accuracy_100 = [0.356588, 0.440528, 0.535970, 0.613763];
accuracy_500 = [0.483668, 0.572416, 0.689424, 0.791391];
accuracy_1000 = [0.563374, 0.720562, 0.762578, 0.836415];
accuracy_3000 = [0.533009, 0.723357, 0.836188, 0.834604];

% ����ʱ�����ݣ�ת��Ϊ���ӣ�
runtime_100 = [51+11/60, 52+35/60, 52+36/60, 54+23/60];
runtime_500 = [14+13/60, 12+48/60, 14+41/60, 42+02/60];
runtime_1000 = [7+57/60, 7+24/60, 8+05/60, 35+01/60];
runtime_3000 = [3+43/60, 5+35/60, 11+20/60, 15+19/60];

% ���䳤��
para_lengths = [100, 500, 1000, 3000];

%% 2. ����׼ȷ�����������仯����
figure;
hold on;
plot(topics, accuracy_100, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '100 ��');
plot(topics, accuracy_500, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '500 ��');
plot(topics, accuracy_1000, '-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '1000 ��');
plot(topics, accuracy_3000, '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '3000 ��');

% ����ͼ���ʽ
xlabel('LDA ������');
ylabel('����׼ȷ��');
title('��ͬ�������µķ���׼ȷ�ʶԱ�');
legend('Location', 'southeast');
grid on;
hold off;

%% 3. ��������ʱ�����������仯����
figure;
hold on;
plot(topics, runtime_100, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '100 ��');
plot(topics, runtime_500, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '500 ��');
plot(topics, runtime_1000, '-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '1000 ��');
plot(topics, runtime_3000, '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '3000 ��');

% ����ͼ���ʽ
xlabel('LDA ������');
ylabel('����ʱ�� (����)');
title('��ͬ�������µ�����ʱ��Ա�');
legend('Location', 'northeast');
grid on;
hold off;

%% 4. ����׼ȷ������䳤�ȱ仯����
figure;
hold on;
plot(para_lengths, [accuracy_100(1), accuracy_500(1), accuracy_1000(1), accuracy_3000(1)], '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '5 ����');
plot(para_lengths, [accuracy_100(2), accuracy_500(2), accuracy_1000(2), accuracy_3000(2)], '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '10 ����');
plot(para_lengths, [accuracy_100(3), accuracy_500(3), accuracy_1000(3), accuracy_3000(3)], '-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '20 ����');
plot(para_lengths, [accuracy_100(4), accuracy_500(4), accuracy_1000(4), accuracy_3000(4)], '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '50 ����');

% ����ͼ���ʽ
xlabel('���䳤�� (��)');
ylabel('����׼ȷ��');
title('��ͬ���䳤���µķ���׼ȷ�ʶԱ�');
legend('Location', 'southeast');
set(gca, 'XScale', 'log'); % ���������������ʾ����
grid on;
hold off;

%% 5. ��������ʱ������䳤�ȱ仯����
figure;
hold on;
plot(para_lengths, [runtime_100(1), runtime_500(1), runtime_1000(1), runtime_3000(1)], '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '5 ����');
plot(para_lengths, [runtime_100(2), runtime_500(2), runtime_1000(2), runtime_3000(2)], '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '10 ����');
plot(para_lengths, [runtime_100(3), runtime_500(3), runtime_1000(3), runtime_3000(3)], '-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '20 ����');
plot(para_lengths, [runtime_100(4), runtime_500(4), runtime_1000(4), runtime_3000(4)], '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '50 ����');

% ����ͼ���ʽ
xlabel('���䳤�� (��)');
ylabel('����ʱ�� (����)');
title('��ͬ���䳤���µ�����ʱ��Ա�');
legend('Location', 'northeast');
set(gca, 'XScale', 'log'); % ���������������ʾ����
grid on;
hold off;

%% 6. ����3D����ͼչʾ׼ȷ������䳤�Ⱥ��������Ĺ�ϵ
[TOPICS, LENGTHS] = meshgrid(topics, para_lengths);

% ׼��׼ȷ�ʾ���
ACCURACY = [accuracy_100; accuracy_500; accuracy_1000; accuracy_3000];

figure;
surf(TOPICS, LENGTHS, ACCURACY);
xlabel('������');
ylabel('���䳤�� (��)');
zlabel('����׼ȷ��');
title('����׼ȷ������䳤�Ⱥ��������Ĺ�ϵ');
colorbar;
grid on;

%% 7. ����3D����ͼչʾ����ʱ������䳤�Ⱥ��������Ĺ�ϵ
RUNTIME = [runtime_100; runtime_500; runtime_1000; runtime_3000];

figure;
surf(TOPICS, LENGTHS, RUNTIME);
xlabel('������');
ylabel('���䳤�� (��)');
zlabel('����ʱ�� (����)');
title('����ʱ������䳤�Ⱥ��������Ĺ�ϵ');
colorbar;
grid on;

%% 8. ���㲢��ʾ���׼ȷ������
[max_acc, max_idx] = max(ACCURACY(:));
[row, col] = ind2sub(size(ACCURACY), max_idx);
fprintf('���׼ȷ������:\n');
fprintf('���䳤��: %d ��\n', para_lengths(row));
fprintf('������: %d\n', topics(col));
fprintf('׼ȷ��: %.4f\n', max_acc);
fprintf('����ʱ��: %.2f ����\n\n', RUNTIME(row, col));

%% 9. ���㲢��ʾ����Լ۱����ã�׼ȷ��/ʱ��ȣ�
acc_time_ratio = ACCURACY ./ RUNTIME;
[max_ratio, max_idx] = max(acc_time_ratio(:));
[row, col] = ind2sub(size(acc_time_ratio), max_idx);
fprintf('����Լ۱�����(׼ȷ��/ʱ��):\n');
fprintf('���䳤��: %d ��\n', para_lengths(row));
fprintf('������: %d\n', topics(col));
fprintf('׼ȷ��: %.4f\n', ACCURACY(row, col));
fprintf('����ʱ��: %.2f ����\n', RUNTIME(row, col));
fprintf('׼ȷ��/ʱ���: %.4f\n', max_ratio);

%% 10. �����ֺʹ��ڲ�ͬ�������µ�׼ȷ������
wordaccuracy_100 = [0.184152, 0.263938, 0.190494, 0.244035];
wordaccuracy_3000 = [ 0.4918, 0.1431, 0.3222, 0.4389];
figure;
hold on;
plot(topics, accuracy_100, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '100 ��');
plot(topics, wordaccuracy_100, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '100 ��');
plot(topics, accuracy_3000, '-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '3000 ��');
plot(topics, wordaccuracy_3000, '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '3000 ��');

% ����ͼ���ʽ
xlabel('LDA ������');
ylabel('����׼ȷ��');
title('�ֺʹ��ڲ�ͬ�������µ�׼ȷ�ʶԱ�');
legend('Location', 'northeast');
grid on;
hold off;

%% 11. �����ֺʹ��ڲ�ͬ�������µ�����ʱ������
wordruntime_100 = [60+26+45/60, 60+32+41/60, 60+25+6/60, 60+52+10/60];
wordruntime_3000 = [5+33/60, 1+16/60, 22+53/60, 26+40/60];
figure;
hold on;
plot(topics, runtime_100, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '100 ��');
plot(topics, wordruntime_100, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '100 ��');
plot(topics, runtime_3000, '-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '3000 ��');
plot(topics, wordruntime_3000, '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '3000 ��');

% ����ͼ���ʽ
xlabel('LDA ������');
ylabel('����ʱ�� (����)');
title('�ֺʹ��ڲ�ͬ�������µ�����ʱ��Ա�');
legend('Location', 'northeast');
grid on;
hold off;
