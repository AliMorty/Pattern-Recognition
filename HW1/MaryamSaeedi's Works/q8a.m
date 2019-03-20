clc
clear

n=1;
N=1000;
mu = 2;
sigma = [1,2,3];

for i=1:3
    subplot(1,3,i)
    data = normrnd(mu, sigma(i), [n N]);
    histogram(data);
    title(strcat('normal distribution with standard deviation=', int2str(sigma(i))));
    axis([-10 10 0 inf]);
end