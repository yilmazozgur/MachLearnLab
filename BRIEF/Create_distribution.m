clear all %overlapped Patches
clc

for i=11:13
    for subn=4:20
        if (i==13 && subn<=6) || (i==12 && subn<=5) || (i==11 && subn<=4)
        else
        Dim='3D';
        test=power(2,i);
        FileName=strcat('F:\Recurrent Holistic Vision v0.4\MatlabCode\BRIEF\BriefOverlappedPatches\PD_',Dim,'_(',int2str(test),'x',int2str(test),'x',int2str(test),')tests_[',int2str(subn),'x',int2str(subn),']size.mat');
        distribution3(subn,test,FileName,Dim);
        end
    end
end


