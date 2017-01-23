function D = rel_entropy_true(p, q)
%rel_entropy_true computes KL-divergence (relative entropy) D(p||q) in bits
%                 for the input distributions

% This function returns a scalar entropy when the input distributions p and
% q are vectors of probability masses, or returns in a row vector the 
% columnwise relative entropies of the input probability matrices p and q.

% Error checking
p0 = p(:);
q0 = q(:);
if (any(size(p) ~= size(q)))
    error('p and q must be equal sizes.');
elseif any(imag(p0)) || any(isinf(p0)) || any(isnan(p0)) || any(p0<0) || any(p0>1)
    error('The probability elements of p must be real numbers between 0 and 1.');
elseif any(imag(q0)) || any(isinf(q0)) || any(isnan(q0)) || any(q0<0) || any(q0>1)
    error('The probability elements of q must be real numbers between 0 and 1.');
elseif any(abs(sum(p)-1) > sqrt(eps))
    error('Sum of the probability elements of p must equal 1.');
elseif any(abs(sum(q)-1) > sqrt(eps))
    error('Sum of the probability elements of q must equal 1.');
end
D = sum(log2(p.^p./q.^p));
end