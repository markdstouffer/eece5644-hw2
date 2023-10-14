load test.mat;

likelihood_ratios = zeros(1, size(x, 2)); 

for i = 1:size(x, 2)
    data_point = x(:, i); 
    likelihood_ratio = (mvnpdf(data_point, [2; 2], [1, 0; 0, 1])) / (0.5 * mvnpdf(data_point, [3; 0], [2, 0; 0, 1]) + 0.5 * mvnpdf(data_point, [0; 3], [1, 0; 0, 2]));
    likelihood_ratios(i) = likelihood_ratio;
end

thresholds = linspace(0, 1, 10000);

total_pos = sum(labels == 1);
total_neg = sum(labels == 0);

true_rate = zeros(size(thresholds));
false_rate = zeros(size(thresholds));

for i = 1:length(thresholds)
    threshold = thresholds(i);

    true_positives = sum(likelihood_ratios >= threshold & labels == 1);
    false_positives = sum(likelihood_ratios >= threshold & labels == 0);
    
    true_rate(i) = true_positives / total_pos;
    false_rate(i) = false_positives / total_neg;
end

thr_optimal = 0.35 / 0.65;

tpr_optimal = sum(likelihood_ratios >= thr_optimal & labels == 1) / total_pos;
fpr_optimal = sum(likelihood_ratios >= thr_optimal & labels == 0) / total_neg;

estimated_labels = (likelihood_ratios >= thr_optimal);
confusion_matrix = confusionmat(labels, estimated_labels);
true_positives = confusion_matrix(2, 2);
true_negatives = confusion_matrix(1, 1);
false_positives = confusion_matrix(1, 2);
false_negatives = confusion_matrix(2, 1);

min_error = (false_positives + false_negatives) / length(labels);

min_err_emp = Inf;
optimal_thresh_emp = 0;

for i = 1:length(thresholds)
    t = thresholds(i);
    err_0 = sum(likelihood_ratios >= t & labels == 0) / sum(labels == 0);
    err_1 = sum(likelihood_ratios < t & labels == 1) / sum(labels == 1);
    err = err_0 * 0.65 + err_1 * 0.35;
    
    if err < min_err_emp
        min_err_emp = err;
        optimal_thresh_emp = t;
    end
end

plot(false_rate, true_rate), hold on,
plot(fpr_optimal, tpr_optimal, 'ro', 'MarkerSize', 10); 
xlabel("FPR");
ylabel("TPR");
title("ROC");
disp(min_error);
disp(min_err_emp);