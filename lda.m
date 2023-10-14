load test.mat x labels;

class1 = x(:, labels == 0);
class2 = x(:, labels == 1);

mu1 = mean(class1, 2);
mu2 = mean(class2, 2);

s1 = cov(class1');
s2 = cov(class2');
sw = s1 + s2;
sb = (mu1 - mu2) * (mu1 - mu2)';

invSw = inv(sw);
invSw_by_sb = invSw * sb;

[V,D] = eig(invSw_by_sb);
[~, idx] = max(diag(D));
W = V(:,idx);

thresholds = linspace(min(W' * x), max(W' * x), 10000);

roc_pts = zeros(length(thresholds), 2);

min_P_Error = Inf;
optimal_threshold = 0;

for i = 1:length(thresholds)
    t = thresholds(i);
    decision = (W' * x >= t);

    TP = sum(decision == 1 & labels == 1);
    TN = sum(decision == 0 & labels == 0);
    FP = sum(decision == 1 & labels == 0);
    FN = sum(decision == 0 & labels == 1);

    TPR = TP / (TP + FN);
    FPR = FP / (FP + TN);

    roc_pts(i, 1) = FPR;
    roc_pts(i, 2) = TPR;
    
    P_Error = (FP + FN) / length(labels);

    if P_Error < min_P_Error
        min_P_Error = P_Error;
        optimal_threshold = t;
    end
end

total_pos = sum(labels == 1);
total_neg = sum(labels == 0);
TPR_min_err = sum(W' * x >= optimal_threshold & labels == 1) / total_pos;
FPR_min_err = sum(W' * x >= optimal_threshold & labels == 0) / total_neg;


plot(roc_pts(:, 1), roc_pts(:, 2)), hold on,
plot(FPR_min_err, TPR_min_err, 'ro', 'MarkerSize', 5);
xlabel('FPR');
ylabel('TPR');
title('ROC using LDA classifier');
disp(min_P_Error);

