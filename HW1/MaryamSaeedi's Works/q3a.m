clc
close all
clear
load iris.dat;

feature = [];
for i=1:3
    feature = [feature,find(iris(:,5)==i)];
end

color={'b*','r*','g*'};


% one feature
for i=1:4
    subplot(2,2, i)
    for j=1:3
        axis([0 80 0 8]);
        plot( iris(feature(:,j), i),j*2, color{j});
        set(gca, 'ytick', []);
        grid on
        hold on
    end
    title(strcat('feature ',int2str(i)));
end
