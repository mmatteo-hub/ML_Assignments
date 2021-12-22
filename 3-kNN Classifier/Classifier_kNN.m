clear;
clc;
close all;
addpath('./mnist');

%% Load data
[X_train, T_train] = loadMNIST(0,0:9);
[X_test, T_test] = loadMNIST(1,0:9);

%% Testing the classifier for differnt Ks
K_array = [1:5,9,16,25]; %[1,2,3,4,5,10,15,20,30,40,50];
accuracy = zeros(10,length(K_array));

for cl=1:10
    T_train_cl = double(T_train == cl);
    T_test_cl = double(T_test == cl);
    
    % take 5% of the data set
    rows = randperm(60000,floor(0.05*60000));
    X_train_sub = X_train(rows,:);
    T_train_sub = T_train(rows,:);
   
    % take 15% of the data set
    rows = randperm(10000,floor(0.15*10000));
    X_test_sub = X_test(rows,:);
    T_test_sub = T_test(rows,:);
    
    
    i=1;
    for j=1:length(K_array)
        [target, err_rate] = kNN(X_train_sub,T_train_sub,X_test_sub,K_array(j),T_test_sub);
        accuracy(cl,i) = 1-err_rate;
        i=i+1;
    end
end

figure;
hold on;
plot(K_array,accuracy(10,:),'DisplayName','Class 0', 'LineWidth', 2);
for i=1:9
    plot(K_array,accuracy(i,:),'LineWidth', 2);
end
title('Accuracy');
xlabel('K');
ylabel('Accuracy in percentage');
legend('Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9')
grid;

lgd = legend;

figure;
bar(1-accuracy);
title('Error computed for each class, on x axis, for each number of neighbours required (each set)');
xlabel('Classes');
ylabel('Error');
legend('Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9','Class 10');
