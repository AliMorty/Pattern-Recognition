
clc
close all
clear
load iris.dat;

feature = [];
for i=1:3
    feature = [feature,find(iris(:,5)==i)];
end

color={'b.','r.','g.'};

% three feature
for i1=1:4
    for i2=i1+1:4
        for i3=i2+1:4
            figure
            for j=1:3
                axis([0 80 0 80 0 80]);
                plot3( iris(feature(:,j), i1),iris(feature(:,j), i2), iris(feature(:,j), i3), color{j});
                xlabel(strcat('feature ', int2str(i1)));
                ylabel(strcat('feature ', int2str(i2)));
                zlabel(strcat('feature ', int2str(i3)));
                grid on
                hold on
            end
            title(strcat('feature ',int2str(i1), ' , ', int2str(i2), ' , ', int2str(i3)));
        end
    end
end