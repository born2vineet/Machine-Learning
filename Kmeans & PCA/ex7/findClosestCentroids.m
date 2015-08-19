function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i = 1:length(X)  % loop over every training example 
    deltas = zeros(K,1); % Set distance between xi and cj = 0
    x = X(i,:);   % x = ith training example
    for j = 1:K   % go over every index of cluster centroid
        k = centroids(j,:);   % k = jth index of cluster centroid 
        delta = x-k;   % compute xi - cj
        deltas(j) = delta * delta';
    end
    [y, idx(i)] = min(deltas); % return the index of centroid which has minimum value of (xi - cj)
end


% =============================================================

end

