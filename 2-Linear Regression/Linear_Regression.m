clc;
close all;

%% Task 1
%% Initialising data

load('turkish-se-SP500vsMSCI.csv');

%Use readmatrix and then remove the first column

mtcarsdata = readmatrix('mtcarsdata-4features.csv');
% Remove the first column because it is a literal colums so data contains in this colum are not usable by MATLAB
mtcarsdata(:,1) = [];

% percentage is the only value choosen by the user
percentage = 0.1;

%% Task 2
%% One dimension linear regression without the Intercept
[x,y] = oneDim_noIntercept(turkish_se_SP500vsMSCI); % no need to take the intercept because the mean value is zero

subplot(2,2,1);
hold on;
plot(turkish_se_SP500vsMSCI(:,1),turkish_se_SP500vsMSCI(:,2),'.','LineWidth',2);
plot(x,y,'LineWidth',2);
xlabel('x values');
ylabel('y values');
legend('Entire data set','Linear regression');
title('One dimension without the Intercept')

%% Creation of a sub set of dimension x% of the original one
[subSet] = subSetCreation(turkish_se_SP500vsMSCI,percentage);

%% One dimension linear regression of a random subset with x% the dimension of the original one
[x_s,y_s] = oneDim_noIntercept(subSet);

subplot(2,2,2);
hold on;
plot(turkish_se_SP500vsMSCI(:,1),turkish_se_SP500vsMSCI(:,2),'.','LineWidth',2);
plot(x,y,'LineWidth',2);
plot(subSet(:,1),subSet(:,2),'.','LineWidth',2);
plot(x_s,y_s,'LineWidth',2);
xlabel('x values');
ylabel('y values');
legend('Data set points','Linear regression','Sub set points','Linear regression of the subset')
title(['Comparison with sub set of dimension ' num2str(percentage*100) '% of the total'])

%% One dimensiom linear regression with Intercept
[x_i,y_i,y_i_c] = oneDim_withIntercept(mtcarsdata);
subplot(2,2,3);
hold on;
plot(x_i,y_i,'.','LineWidth',2);
plot(x_i,y_i_c,'LineWidth',2);
xlabel('mpg (miles per gallon)');
ylabel('weigth');
legend('Data set','Linear regression with intercept');
title('Linear regression with Intercept');

%% Multi dimensional problem
[y_m,t] = multi_Dim(mtcarsdata);
subplot(2,2,4);
hold on;
plot(t,y_m,'.','LineWidth',2);
plot(y_m,y_m,'-','LineWidth',2);
legend('Approssimation of the diagonal of the square','Diagonal of the square');
title('Multi variable regression model');

%% Task 3
%% Point 1, 2 and 4 with 5% of the total as train and 95% as test
percentage2 = 0.05;
len = 10;
err_test1 = zeros(len,1);
err_train1 = zeros(len,1);
for k=1:len
[subSet1_5,subSet1_95] = setSplit(turkish_se_SP500vsMSCI,percentage2);
[x,y] = oneDim_noIntercept(subSet1_5);
w = y\x;
target1_95 = subSet1_95(:,2);
y_calc1_95 = w * subSet1_95(:,1);
err_test1(k) = immse(y_calc1_95,target1_95);
target1_5 = subSet1_5(:,2);
y_calc1_5 = w * subSet1_5(:,1);
err_train1(k) = immse(y_calc1_5,target1_5);
end
mse1 = [err_train1 err_test1];

figure
subplot(1,3,1);
diag = bar(mse1);
diag(1).FaceColor = 'g';
diag(2).FaceColor = 'r';
ylabel('MSE');
legend('MSE train set','MSE test set')
title(['MSE comparison between test and train set among ' num2str(len) ' random iterations'])


err_test3 = zeros(len,1);
err_train3 = zeros(len,1);

for j=1:len
[subSet2_5,subSet2_95] = setSplit(mtcarsdata,percentage2);
[xn,yn,yn_c] = oneDim_withIntercept(subSet2_5);
x_sub_95 = subSet2_95(:,4);
y_sub_95 = subSet2_95(:,1);
w1_95 = (sum((x_sub_95-mean(x_sub_95)).*(y_sub_95-mean(y_sub_95))))./(sum((x_sub_95-mean(x_sub_95)).^2));
w0_95 = mean(y_sub_95) - w1_95 * mean(x_sub_95);
y_calc2_95 = w0_95 + w1_95 .* x_sub_95;
err_test3(j) = immse(y_calc2_95,y_sub_95);
err_train3(j) = immse(yn_c,yn);
end
mse3 = [err_train3 err_test3];

subplot(1,3,2);
diag = bar(mse3);
diag(1).FaceColor = 'g';
diag(2).FaceColor = 'r';
ylabel('MSE');
legend('MSE train set','MSE test set')
title(['MSE comparison between test and train set among ' num2str(len) ' random iterations'])


err_test4 = zeros(len,1);
err_train4 = zeros(len,1);

for h=1:len
[subSet4_5,subSet4_95] = setSplit(mtcarsdata,percentage2);
[y4,t4,w4] = multi_Dim(subSet4_5);
target4_95 = subSet4_95(:,1);
x_95 =subSet4_95(:,2:end);
w = (pinv(x_95'*x_95))*x_95'*target4_95;
y_calc4_95 = x_95 * w;
err_test4(h) = immse(y_calc4_95,target4_95);
err_train4(h) = immse(y4,t4);
end
mse4 = [err_train4 err_test4];

subplot(1,3,3);
diag = bar(mse4);
diag(1).FaceColor = 'g';
diag(2).FaceColor = 'r';
ylabel('MSE');
legend('MSE train set','MSE test set')
title(['MSE comparison between test and train set among ' num2str(len) ' random iterations'])









