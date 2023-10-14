function [x, labels] = generateDataQ2()
    N = 10000;
    classPriors = [0.3, 0.3, 0.4];
    labels = zeros(1, N);

    for i = 1:N
        r = rand();
        if r <= classPriors(1)
            labels(i) = 1;
        elseif r <= classPriors(1) + classPriors(2)
            labels(i) = 2;
        else
            labels(i) = 3;
        end
    end

    for l = 1:3
        indl = find(labels == l);
        if l == 1
            N1 = length(indl);
            m1 = [3 4 3];
            C1 = eye(3);
            x(:, indl) = mvnrnd(m1, C1, N1)';
        elseif l == 2
            N2 = length(indl);
            m2 = [5 2 4];
            C2 = eye(3);
            x(:, indl) = mvnrnd(m2, C2, N2)';
        else
            N3 = length(indl);
            w3 = [0.5, 0.5];
            mu3 = [4 3; 3 5; 2 1];
            Sigma3(:,:,1) = eye(3);
            Sigma3(:,:,2) = eye(3);
            gmmParameters.priors = w3;
            gmmParameters.meanVectors = mu3;
            gmmParameters.covMatrices = Sigma3;
            [x(:, indl), components] = generateDataFromGMM(N3, gmmParameters);
        end
    end


    [conf, decision] = estimateConfusionMatrix(x, labels);
    plot3DScatter(x, labels, decision);
    file_name = 'q2data.mat';
    save(file_name, 'x', 'labels', 'conf', 'decision');
    

function [x,labels] = generateDataFromGMM(N,gmmParameters)
% Generates N vector samples from the specified mixture of Gaussians
% Returns samples and their component labels
% Data dimensionality is determined by the size of mu/Sigma parameters
priors = gmmParameters.priors; % priors should be a row vector
meanVectors = gmmParameters.meanVectors;
covMatrices = gmmParameters.covMatrices;
n = size(gmmParameters.meanVectors,1); % Data dimensionality
C = length(priors); % Number of components
x = zeros(n,N); labels = zeros(1,N); 
% Decide randomly which samples will come from each component
u = rand(1,N); thresholds = [cumsum(priors),1];
for l = 1:C
    indl = find(u <= thresholds(l)); Nl = length(indl);
    labels(1,indl) = l*ones(1,Nl);
    u(1,indl) = 1.1*ones(1,Nl); % these samples should not be used again
    x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
end

function plot3DScatter(x, labels, decision)
    marker_shapes = {'o', 's', '^'};

    correct_color = 'g';
    incorrect_color = 'r';

    figure;
    disp(size(x));
    for i = 1:size(x, 2)
        data_point = x(:, i);

        true_class = labels(i);
        predicted_class = decision(i);

        marker_shape = marker_shapes{true_class};

        if true_class == predicted_class
            marker_color = correct_color;
        else
            marker_color = incorrect_color;
        end

        scatter3(data_point(1), data_point(2), data_point(3), 'filled', marker_shape, 'MarkerEdgeColor', marker_color);
        hold on;
    end

    % Set axis labels
    xlabel('X-axis');
    ylabel('Y-axis');
    zlabel('Z-axis');

    % Add a legend
    legend('Class 1', 'Class 2', 'Class 3');


function [confusionMatrix, decision] = estimateConfusionMatrix(x, labels)
   confusionMatrix = zeros(3, 3);
   classPriors = [0.3, 0.3, 0.4];
   decision = zeros(1, size(x, 2));

   for i = 1:size(x, 2)
    sample = x(:, i);
    trueLabel = labels(i);

    posteriorProbabilities = zeros(1, 3);
    for j = 1:3
        if j == 1
            mu = [3 4 3];
            Sigma = eye(3);
        elseif j == 2
            mu = [5 2 4];
            Sigma = eye(3);
        else
            mu1 = [4 3 2];
            mu2 = [3 5 1];
            if rand() > 0.5
                mu = mu1;
            else
                mu = mu2;
            end
            Sigma = eye(3);
        end

        likelihood = mvnpdf(sample', mu, Sigma);
        posteriorProbabilities(j) = likelihood * classPriors(j);
    end
    
        [~, d] = max(posteriorProbabilities);
        decision(i) = d;
        
        confusionMatrix(trueLabel, d) = confusionMatrix(trueLabel, d) + 1;

        
    end