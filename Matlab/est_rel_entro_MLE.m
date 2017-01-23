function est = est_rel_entro_MLE(sampP, sampQ)
%est_rel_entro_MLE   Maximum likelihood estimate of Kullback-Leibler
%                    divergence or relative entropy (in bits) of the input
%                    sample
%
% This function returns a scalar MLE of the KL divergence between
% sampP and sampQ when they are vectors, or returns a row vector consisting
% of the MLE of each column of the samples when they are matrices.
%
% Input:
% ----- sampP: a vector or matrix which can only contain integers. The input
%              data type can be any interger classes such as uint8/int8/
%              uint16/int16/uint32/int32/uint64/int64, or floating-point
%              such as single/double.
% ----- sampQ: same conditions as sampP. Must have the same number of
%              columns if a matrix.
% Output:
% ----- est: the KL divergence (in bits) of the input vectors or that of
%            each column of the input matrices. The output data type is
%            double.

if ~isequal(sampP, fix(sampP)) || ~isequal(sampQ, fix(sampQ))
    error('Input arguments P and Q must only contain integers.');
end
if isrow(sampP)
    sampP = sampP.';
end
if isrow(sampQ)
    sampQ = sampQ.';
end
[m,sizeP] = size(sampP);
[n,seq_num] = size(sampQ);
if sizeP ~= seq_num
    error('Input arguments P and Q must have the same number of columns');
end

% empirical distros + fingerprints

% Map non-consecutive integer samples to consecutive integer numbers
% along each column of X and Y (which start with 1 and end with the
% total number of distinct samples in each column). For example,
%                 [  1    6    4  ]        [ 1  3  3 ]
%                 [  2    6    3  ] -----> [ 2  3  2 ]
%                 [  3    2    2  ]        [ 3  1  1 ]
%                 [ 1e5   3   100 ]        [ 4  2  4 ]
% The purpose of this mapping is to shrink the effective data range to
% avoid possible numerical overflows.

[PQ_len, PQ_wid] = size([sampP;sampQ]);
[PQ_seq, id] = sort([sampP;sampQ]);
PQ_seq(bsxfun(@plus, id, (0:PQ_wid-1)*PQ_len)) = ...
    cumsum([ones(1,PQ_wid,'int64');sign(diff(PQ_seq))]);
S = max(max(PQ_seq));
sampP = PQ_seq(1:m, :);
sampQ = PQ_seq(m+1:end, :);

e_p = histc(sampP,1:S);
e_q = histc(sampQ,1:S);
if isrow(e_p)
    e_p = e_p';
end
if isrow(e_q)
    e_q = e_q';
end
    
bins = max([max(e_p),max(e_q)]);
prob_q = (0:bins)'/n;
prob_mat = log_mat_MLE(prob_q, n, seq_num);

sum_p = zeros(size(prob_mat));
for row_iter = (unique(e_q)+1)'
    sum_p(row_iter,:) = sum(e_p.*(e_q==row_iter-1),1)/m;
end
d = sum(sum_p.*prob_mat)/log(2);
entro = est_entro_MLE(sampP);
est = max(0, -entro - d);

end

function output = log_mat_MLE(x, n, seq_num)
    X = repmat(x,1,seq_num);
    output = log(X);
    output(X==0) = -log(n);
end

function est = est_entro_MLE(samp)
%est_entro_MLE  Maximum likelihood estimate of Shannon entropy (in bits) of 
%               the input sample
%
% This function returns a scalar MLE of the entropy of samp when samp is a 
% vector, or returns a (row-) vector consisting of the MLE of the entropy 
% of each column of samp when samp is a matrix.
%
% Input:
% ----- samp: a vector or matrix which can only contain integers. The input
%             data type can be any interger classes such as uint8/int8/
%             uint16/int16/uint32/int32/uint64/int64, or floating-point 
%             such as single/double. 
% Output:
% ----- est: the entropy (in bits) of the input vector or that of each 
%            column of the input matrix. The output data type is double. 


    if ~isequal(samp, fix(samp))
        error('Input sample must only contain integers.');
    end

    if isrow(samp)
        samp = samp.';
    end
    [n, wid] = size(samp);

%     % A fast algorithm to compute the column-wise histogram of histogram (fingerprint) 
%     f = find([diff(sort(samp))>0; true(1,wid)]);  % Returns in column vector f the linear indices to the last occurrence of repeated values along every column of samp
%     f = accumarray({filter([1;-1],1,f),ceil(f/n)},1)   % f: fingerprint   

    % A memory-efficient algorithm for computing fingerprint when wid is large, e.g., wid = 100
    d = [true(1,wid);logical(diff(sort(samp),1,1));true(1,wid)];
    for k = wid:-1:1
        a = diff(find(d(:,k)));
        id = 1:max(a);  
        f(id,k) = histc(a,id);
    end

    prob = (1:size(f,1))/n;
    prob_mat = -prob.*log2(prob);
    est = prob_mat * f;
end