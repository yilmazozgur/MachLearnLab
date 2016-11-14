clear; clc;
for i=2:40

tests=128;
imgsize=[i*4 2*i];

center=[imgsize(1)/2 imgsize(2)/2];

% S=imgsize(1)/2; sigma=S/5;
sigma=imgsize(1)/5;
pd=makedist('Normal','mu',0,'sigma',sigma);
% xy_pair=zeros(NumberOfTests*4,1);
x_points=zeros(tests*2,1); y_points=zeros(tests*2,1);

for i=1:tests*2
    x_points(i)=abs(imgsize(1)-abs(round(random(pd))+center(1)));
end

for i=1:tests*2
    y_points(i)=abs(imgsize(2)-abs(round(random(pd))+center(2)));
end



SavePath='/home/comp2/Desktop/BRIEF';
FileName=fullfile(SavePath,strcat('PD_',int2str(tests),'tests_[',int2str(imgsize(1)),'x',int2str(imgsize(2)),']size'));
save(FileName,'x_points','y_points');

    for i=1:tests
        x=[x_points(i), x_points(tests*2-i)];
        y=[y_points(i), y_points(tests*2-i)];
        plot(x,y); hold on;
        axis([0 imgsize(1) 0 imgsize(2)]);
    end
    savefig(h,'figure.fig');

end