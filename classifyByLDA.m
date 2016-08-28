function [ Y ] = classifyByLDA( Gmd, X )
% CLASSIFYBYLDA classify by a already trained LDA classifier. 
% Usage:
% [y] = classifyByLDA(Gmd, testdata)
% Inputs:
%       -Gmd: struct contains mean values and covariance matrix of
%       gaussian distribution of different classes
%       -X: test data matrix, n-by-d matrix, n is number of samples
%       and d is dimension after projection
% Outputs:
%       -Y: results of classification
%
% @Author: HammerZhang
% @Time: 2016.7.23 21:26
% 
% ========================================================================

% load gaussian distribution parameters
u       = Gmd.U;
sigma   = Gmd.C;
p       = Gmd.P;

cn      = size(u,2);                        % number of classes
len     = size(X,1);                        % length of data

for ci = 1:cn
    tmp  = 1 / (2*pi*det(sigma(:,:,ci)))^0.5 * exp(-0.5*(X-repmat(u(:,ci)',1,len)...,
        * sigma(:,:,ci)*(X'-repmat(u(:,ci),len,1))));
    Px_k(:,ci) = diag(tmp);
end

% posterior probability
Pk_x    = Px_k .* repmat(P,len,1);
[~,I]   = max(Pk_x,[],2);
Y       = zeros(1,len);
Y(I)    = 1;


end

