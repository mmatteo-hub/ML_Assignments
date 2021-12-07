clear;
clc;
close all;
addpath('./mnist');

%% Load data
[X_train, T_train] = loadMNIST(0,0:9);
[X_test, T_test] = loadMNIST(1,0:9);

%% Testing the classifier for differnt Ks
K_array = [1,2,3,4,5,10,15,20,30,40,50];
accuracy = zeros(10,length(K_array));

for cl=1:10
    T_train_cl = double(T_train == cl);
    T_test_cl = double(T_test == cl);
    
    % take 5% of the data set
    rows = randperm(60000,floor(0.025*60000));
    X_train_sub = X_train(rows,:);
    T_train_sub = T_train(rows,:);
    
    % take 10% of the data set
    rows = randperm(10000,floor(0.10*10000));
    X_test_sub = X_test(rows,:);
    T_test_sub = T_test(rows,:);
    
    
    for i=1:length(K_array)
        for j=1:length(K_array)
            [target, err_matrix] = kNN(X_train_sub,T_train_sub,X_test_sub,K_array(j),T_test_sub);
            accuracy(cl,i) = 1-err_matrix;
        end
    end
end

figure;
hold on;
plot(K_array,accuracy(10,:),'DisplayName','Class 0', 'LineWidth', 2);
plot(K_array,accuracy(1,:),'DisplayName','Class 1', 'LineWidth', 2);
plot(K_array,accuracy(2,:),'DisplayName','Class 2', 'LineWidth', 2);
plot(K_array,accuracy(3,:),'DisplayName','Class 3', 'LineWidth', 2);
plot(K_array,accuracy(4,:),'DisplayName','Class 4', 'LineWidth', 2);
plot(K_array,accuracy(5,:),'DisplayName','Class 5', 'LineWidth', 2);
plot(K_array,accuracy(6,:),'DisplayName','Class 6', 'LineWidth', 2);
plot(K_array,accuracy(7,:),'DisplayName','Class 7', 'LineWidth', 2);
plot(K_array,accuracy(8,:),'DisplayName','Class 8', 'LineWidth', 2);
plot(K_array,accuracy(9,:),'DisplayName','Class 9', 'LineWidth', 2);
title('Accuracy');
xlabel('K');
ylabel('Accuracy in percentage');
grid;

lgd = legend;

