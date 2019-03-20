clc
clear
close all

N=1000;
mu = [1 2];
cov = [3 1; 1 2];
data = mvnrnd(mu, cov, N);
% contour(data)
Z = mvnpdf(data,mu,cov); %// compute Gaussian pdf
% Z = reshape(Z,size(data)); %// put into same size as X, Y
contour(Z); %, axis equal  %// contour plot; set same scale for x and y...