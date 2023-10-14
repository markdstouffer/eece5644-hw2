load('q2data.mat');

L10 = [0 1 10; 1 0 10; 1 1 0];
L100 = [0 1 100; 1 0 100; 1 1 0];

confusionMatrixL10 = zeros(numClasses, numClasses);
confusionMatrixL100 = zeros(numClasses, numClasses);

classPriors = [0.3, 0.3, 0.4];

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
    
    [~, decisionL10] = min(posteriorProbabilities * L10);

    [~, decisionL100] = min(posteriorProbabilities * L100);

    confusionMatrixL10(trueLabel, decisionL10) = confusionMatrixL10(trueLabel, decisionL10) + 1;
    confusionMatrixL100(trueLabel, decisionL100) = confusionMatrixL100(trueLabel, decisionL100) + 1;
end

figure;
subplot(1, 2, 1);
imagesc(confusionMatrixL10);
title('L10 Confusion Matrix');
xlabel('Predicted');
ylabel('Actual');
colorbar;

subplot(1, 2, 2);
imagesc(confusionMatrixL100);
title('L100 Confusion Matrix');
xlabel('Predicted');
ylabel('Actual');
colorbar;
