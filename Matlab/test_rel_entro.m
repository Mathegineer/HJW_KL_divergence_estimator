clear; clc; close all;

uS = 1 + 20 * rand(1);
cM = 1;
cN = 4;
num = 30;
mc_times = 20;  % Total number of Monte-Carlo trials for each alphabet size
record_S = ceil(logspace(1,5,num));
record_m = ceil(cM*record_S./log(record_S));
record_n = ceil(cN*uS*record_S./log(record_S));
record_HJW = zeros(num, mc_times);
record_true = zeros(num, mc_times);
record_MLE  = zeros(num, mc_times);
twonum = rand(2,1);
for iter = num:-1:1
    S = record_S(iter);
    m = record_m(iter);
    n = record_n(iter);
    dist1 = betarnd(twonum(1),twonum(2),S,1);
    dist1 = dist1/sum(dist1);
    ratio = 1 + (uS-1).*rand(size(dist1));
    dist2 = dist1./ratio;
    dist2(end) = 1 - sum(dist2(1:end-1));
    samp1 = randsmpl(dist1, m, mc_times, 'int32');
    samp2 = randsmpl(dist2, n, mc_times, 'int32');
    record_true(iter,:) = repmat(rel_entropy_true(dist1,dist2),1,mc_times);
    record_HJW(iter,:) = est_rel_entro_HJW(samp1,samp2);
    record_MLE(iter,:) = est_rel_entro_MLE(samp1,samp2);
end

HJW_err = sqrt( mean((record_HJW - record_true).^2,2) );
MLE_err = sqrt( mean((record_MLE - record_true).^2,2) );
figure
plot(record_S*uS./record_n, HJW_err,'b-s','LineWidth',2,'MarkerFaceColor','b'); hold on;
plot(record_S*uS./record_n, MLE_err, 'r-.o','LineWidth',2,'MarkerFaceColor','r');
xlabel('Su(S)/n')
legend('HJW','MLE','location','northwest');
ylabel('Root Mean Squared Error')
title('Relative Entropy Estimation')