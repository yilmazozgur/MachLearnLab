clear all; clc;
% Color={'RGB','YIQ','Gray'}; %'YCbCr','HSV'
% Sumt={'Coarse','Fine','Finest','Single'};


Epoch=100; %50:10:60 % Epoch Number Parameter
%     iter=3;
    

    for l=3:3
%         for patch=4:12
            AutoExperimentClassification_MemoryEff(Epoch,power(10,l-4));
            clc;
%         end
    end
       
    
    
    
    
%     
%     optionsAll{1}=optionsData;
% optionsAll{2}=optionsExtractFeatures;
% optionsAll{3}=optionsPoolFeatures;
% optionsAll{4}=optionsNormalizeBinarize;
% optionsAll{5}=optionsExpansion;
% optionsAll{6}=optionsSupervisedLearning;
% optionsAll{7}=optionsEvaluation;
% optionsAll{8}=optionsKmeans;
% optionsAll{9}=optionsBriefFeatureExtraction;
% optionsAll{10}=RootPath;
    
    
    
    
    
    
    
    

    
    
    
    
    
    
%     for iter=3 %2:4 %3:6 % NumberOfMax Iteration parameter
%         
%         for l=2:5 %%lambda parameter (1e-3..1e+3)          
%             for i=7:9 % Number Of Tests Parameter; 128,256,512,1024,2048
%                 for sm=1:4 % SummationType parameter
%                     if ((sm==3)&&(i>=9)) || ((sm==2)&&(i>=10)) %this if check prevents occuring the error "GPU memory is not sufficient"
%                         continue;
%                     end
%                     
% %                     if (iter==2) && (l==1) && (i==7) && ((sm==1)||(sm==2))
% %                         continue;
% %                     end % to skip already calculated parameters;
%                       for c=1:3 %Color Space Parameter
%                           
%                         opt=AutoParametersClassification_Alisher(Color{c},   Sumt{sm},     Epoch,  iter,   power(2,i),power(10,l-3),-100);
%                         %                                        ColorSpcae, SummationType,EpochNo,MaxIter,Bits,      lambda,       CVthreshold                   
%                         AutoExperimentClassification_MemoryEff(opt);
%                         clc;                              
%                       end
%                         fid = fopen('/home/comp2/Desktop/Results.txt','a+');
%                         fprintf(fid,'-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n');
%                         fclose(fid);
%                 end
%             end
%         end
%     end


