function est = est_rel_entro_HJW(sampP, sampQ)
% est = est_rel_entro(sampP, sampQ)
%
% This function returns a scalar estimate of the Kullback-Leibler divergence 
% D(P||Q) between sampP and sampQ when they are vectors, or returns a row 
% vector containing the estimate of each column of the samples when they 
% are matrices.
%
% Input:
% ----- sampP: a vector or matrix which can only contain integers. The input
%              data type can be any integer classes such as uint8/int8/
%              uint16/int16/uint32/int32/uint64/int64, or floating-point 
%              such as single/double. 
% ----- sampQ: same conditions as sampP. Must have the same number of
%              columns if a matrix.
% Output: 
% ----- est: the Kullback-Leibler divergence (in bits) of the input vectors 
%            or that of each column of the input matrices. The output data 
%            type is double.

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
[c_1,MLE_const] = const_gen(n);
c_1 = repmat(c_1,1,seq_num);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
order = min(4+ceil(1.2*log(n)), 21);  % The order of polynomial is no more than 21, otherwise floating-point error occurs
persistent poly_entro;
if isempty(poly_entro)
    load poly_coeff_entro.mat poly_entro;
end
coeff = - poly_entro{order + 1}(2:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
prob_mat = log_mat(prob_q,n,coeff,c_1,MLE_const);

sum_p = zeros(size(prob_mat));
for row_iter = (unique(e_q)+1)'
    sum_p(row_iter,:) = sum(e_p.*(e_q==row_iter-1),1)/m;
end
d = sum(sum_p.*prob_mat)/log(2);
entro = est_entro_JVHW(sampP);
est = max(0, - entro - d);

end

function output = log_mat(x, n, g_coeff, c_1, const)
    K = length(g_coeff) - 1;   % g_coeff = {g1, g2, ..., g_K+1}, K: the order of best polynomial approximation
    thres = 2*c_1*log(n)/n;
    [T, X] = meshgrid(thres,x);
    ratio = min(max(2*X./T-1,0),1);
    q = reshape(0:K-1,1,1,K);
    g = repmat(reshape(g_coeff,1,1,K+1),1,size(c_1,2));
    g(:,:,1) = g(:,:,1) + log(thres);
    MLE = log(X) + (1-X)./(2*X*n);
    MLE(X==0) = -log(n) - const;
    polyApp = sum(bsxfun(@times, cumprod(cat(3, ones(size(T)), bsxfun(@minus, n*X, q)./bsxfun(@times, T, n-q)),3), g), 3);
    polyfail = isnan(polyApp) | isinf(polyApp);
    polyApp(polyfail) = MLE(polyfail);
    output = ratio.*MLE + (1-ratio).*polyApp;
end

function [c_1,const] = const_gen(n)
% const_gen     Generates the optimal c_1 and MLE_const for the relative
%               entropy estimator
% 
% This function returns a size 2 row vector. The first element is the
% optimal c_1. The second element is the optimal MLE adjustment constant.
%
% Input: 
% ----- n: The number of samples from the Q distribution
%
% Output: 
% ----- c_1: optimal c_1
% ----- const: optimal MLE constant, c_1 * const == 0
    const = 0;
    c_1 = 0;
    if log(n) < 3.2
        const = 1;
    elseif log(n) < 3.4
        const = 1.7;
    elseif log(n) < 4.9
        const = 1.7;
    elseif log(n) < 6.9
        const = 1.72;
    elseif log(n) < 7.2
        c_1 = -0.340909+0.272727*log(n);
    elseif log(n) < 9.3
        c_1 = 2.49953-0.122084*log(n);
    elseif log(n) < 11.4
        c_1 = 2.10767-0.0718463*log(n);
    else
        c_1 = 1.07;  
    end
end

function est = est_entro_JVHW(samp)
%est_entro_JVHW   Proposed JVHW estimate of Shannon entropy (in bits) of 
%                 the input sample
%
% This function returns a scalar JVHW estimate of the entropy of samp when 
% samp is a vector, or returns a row vector containing the JVHW estimate of
% each column of samp when samp is a matrix.
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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    order = min(4+ceil(1.2*log(n)), 22);  % The order of polynomial is no more than 22 because otherwise floating-point error occurs
    persistent poly_entro;
    if isempty(poly_entro)
        load poly_coeff_entro.mat poly_entro;
    end
    coeff = poly_entro{order};   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
        
    % Piecewise linear/quadratic fit of c_1
    V1 = [0.3303 0.4679];     
    V2 = [-0.530556484842359,1.09787328176926,0.184831781602259];   
    f1nonzero = f(1,:) > 0;
    c_1 = zeros(1, wid);
    if n >= order && any(f1nonzero)
        if n < 200
            c_1(f1nonzero) = polyval(V1, log(n./f(1,f1nonzero)));   
        else
            n2f1_small = f1nonzero & log(n./f(1,:)) <= 1.5;
            n2f1_large = f1nonzero & log(n./f(1,:)) > 1.5;
            c_1(n2f1_small) = polyval(V2, log(n./f(1,n2f1_small)));  
            c_1(n2f1_large) = polyval(V1, log(n./f(1,n2f1_large)));  
        end
        c_1(f1nonzero) = max(c_1(f1nonzero), 1/(1.9*log(n)));  % make sure nonzero threshold is higher than 1/n
    end
    
    prob_mat = entro_mat(prob, n, coeff, c_1);
    est = sum(f.*prob_mat, 1)/log(2); 
end 


function output = entro_mat(x, n, g_coeff, c_1)
    K = length(g_coeff) - 1;   % g_coeff = {g0, g1, g2, ..., g_K}, K: the order of best polynomial approximation, 
    thres = 4*c_1*log(n)/n;
    [T, X] = meshgrid(thres,x);   
    ratio = min(max(2*X./T-1,0),1);
    q = reshape(0:K-1,1,1,K); 
    g = reshape(g_coeff,1,1,K+1);  
    MLE = -X.*log(X) + 1/(2*n);   
    polyApp = sum(bsxfun(@times, cumprod(cat(3, T, bsxfun(@minus, n*X, q)./bsxfun(@times, T, n-q)),3), g), 3) - X.*log(T);     
    polyfail = isnan(polyApp) | isinf(polyApp); 
    polyApp(polyfail) = MLE(polyfail);   
    output = ratio.*MLE + (1-ratio).*polyApp; 
    output = max(output,0);
end