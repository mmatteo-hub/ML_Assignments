% INPUT

% training_set: matrix [n,d+1]
% test_set: matrix [m,c]

% OUTPUT

% target:
% classification: function g_i obtained
% error_rate: the error computed among the results

function[target,classificationNoLp,classificationLp,error_rate] = NaiveBayesClassifier(training_set, test_set)
%% Definition of matrices dimensions
[row_train,col_train]=size(training_set);
[row_test,col_test]=size(test_set);

%% Check the correct data set's dimensions
if(col_test >= col_train - 1)
    
    %% Check elements different form -1
    for i=1:row_train
        for k=1:col_train
            if(training_set(i,k)==-1)
                disp('Error value in the training set.');
            end
        end
    end
    for i=1:row_test
        for k=1:col_test
            if(test_set(i,k)==-1)
                disp('Error value in the test set.');
            end
        end
    end
    
    %% Check values inside test set equals to the ones in the training set
    row_eliminated = -ones(row_test,1);
    for q=1:row_test
        for w=1:col_test
            b = ismember(test_set(q,w),unique(training_set(:,w)));
            if(b == false)
                row_eliminated(q,1) = q;
            elseif(row_eliminated(q,1) == -1)
                row_eliminated(q,1) = 0;
            end
        end
    end
    for e=1:length(row_eliminated)
        if(row_eliminated(e) ~= 0)
            test_set(row_eliminated(e),:) = [];
            row_test = row_test - 1;
        end
    end
    
    %% Training the Naive Bayes classifier
    
    % Compute the number of classes and which are the classes
    classes = unique(training_set(:,col_train));
    n_classes = length(classes);
    n_variables = col_train - 1;
    
    % we compute the prob of outlook,temp,hum,wind of both classes (to be done
    % in general), then we save al the prob using cells array. Each cell has to
    % have a properly dimension according to the probs calculated
    
    % Initialise the cell array for the probabilities
    probs = cell(n_classes,n_variables);
    probs_Laplace = cell(n_classes,n_variables);
    prob_class = zeros(1,n_classes);
    
    % Calculate the number of values assumed by a variable
    var_arr = zeros(1,n_variables);
    
    % Fill the probabilities cell array with the corrisponding v
    for i=1:n_classes
        numTot = sum(training_set(:,col_train) == classes(i));
        prob_class(1,i) = numTot/row_train;
        for j=1:n_variables
            values = unique(training_set(:,j));
            for k=1:length(values)
                cont = 0;
                for l=1:row_train
                    if(training_set(l,j) == values(k) && training_set(l,col_train) == classes(i))
                        cont = cont + 1;
                    end
                end
                probs{i,j}(k) = cont / numTot;
                % Probabilities computed with Laplace smoothing
                a = 1;
                probs_Laplace{i,j}(k) = (cont + a) / (numTot + a * var_arr(j));
            end
        end
    end
    
    %% COMPUTE THE TARGET
    product = ones(row_test,n_classes);
    product_Laplace = ones(row_test,n_classes);
    g_i = zeros(row_test,n_classes);
    g_i_Laplace = zeros(row_test,n_classes);
    target_no_Lp = -ones(1,row_test);
    target_Laplace = -ones(1,row_test);
    for r=1:row_test
        for cl=1:n_classes
            for col=1:col_test-1
                p_var = probs{cl,col};
                p_var_Laplace = probs_Laplace{cl,col};
                product(r,cl) = product(r,cl)*p_var(1,test_set(r,col));
                product_Laplace(r,cl) = product_Laplace(r,cl)*p_var_Laplace(1,test_set(r,col));
            end
            g_i(r,cl) = prob_class(1,cl)*product(r,cl);
            g_i_Laplace(r,cl) = prob_class(1,cl)*product_Laplace(r,cl);
            
            if col_test == col_train
                if(cl == 1)
                    target_no_Lp(1,r) = cl;
                elseif (g_i(r,cl) > g_i(r,cl-1)) % bring the smallest because we want the minimun error
                    target_no_Lp(1,r)=cl;
                end
            end
            if col_test == col_train
                if(cl == 1)
                    target_Laplace(1,r) = cl;
                elseif (g_i_Laplace(r,cl) > g_i_Laplace(r,cl-1)) % bring the smallest because we want the minimun error
                    target_Laplace(1,r)=cl;
                end
            end
        end
    end
    target = [target_no_Lp;target_Laplace];
    classificationNoLp = g_i;
    classificationLp = g_i_Laplace;
    
    
    %% COMPUTE THE ERROR-RATE
    if (col_test ~= col_train)
        error_rate = -1;
        return;
    end
    
    err = 0;
    err_Laplace = 0;
    for h=1:row_test
        if(test_set(h,col_test) ~= target_no_Lp(1,h))
            err = err + 1;
        end
        if(test_set(h,col_test) ~= target_Laplace(1,h))
            err_Laplace = err_Laplace + 1;
        end
    end
    error_rate = [err/row_test;err_Laplace/row_test];
else
    disp('Test set matrix^s size is not correct.');
end

end