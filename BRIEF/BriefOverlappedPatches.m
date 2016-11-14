function Features=BriefOverlappedPatches(Dataset,ReceptiveFields,DataStats,options) %optionsAll
tic

RootPath=options{10};
WhichData=options{1}.WhichData;
TrainTest=options{1}.TrainTest;
if strcmp(WhichData,'Features') ||  strcmp(WhichData,'ExpandedFeatures') || strcmp(WhichData,'Model')
    Features=Dataset;
    return
end

savedir=strcat(RootPath,'\BRIEF\BriefOverlappedPatches\Saved BRIEF_OverlappedPatches Features\',options{9}.Domain,options{9}.SampleNo,TrainTest,'-');

optionsData      = options{1};                   
optionsExtract   = options{2};
optionsPool      = options{3};                                   
optionsNormalize = options{4};
Domain              = options{9}.Domain;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%if no expansion is needed, return.                                             
%This is necessary not to make a major change in the experiment script.         
if not(optionsExtract.Perform)                                                  
    if strcmp(Dataset.options{1,1}.TrainTest,'Train')                           
        Features.Features=Dataset.trainX;                                       
        Features.Labels=Dataset.trainY;                                         
    else                                                                        
        Features.Features=Dataset.testX;                                        
        Features.Labels=Dataset.testY;                                          
    end                                                                         
    Features.UniqueLabels=Dataset.UniqueLabels;                                 
    Features.optionsAll=options;                                                
    Features.optionsPrev=options;                                                 
    Features.RFData=ReceptiveFields;                                            
    Features.SparsifyFlag=optionsExtract.SparsifyFlag;                          
    Features.SparsityMultiplier=optionsExtract.SparsityMultiplier;              
    Features.SummationType=optionsPool.SummationType;                           
    Features.Binarize=optionsNormalize.Binarize;                                
    Features.DatasetName=Dataset.DatasetName;                                   
                                                                                
    if strcmp(optionsData.TrainTest,'Train')                                    
        if isempty(DataStats)                                                   
            XPooled_mean = mean(Features.Features);                             
            XPooled_sd = sqrt(var(Features.Features)+0.01);                     
        else %for batch based processing, combine with previous batch stats     
            XPooled_mean = (mean(Features.Features)+DataStats.Mean)/2;          
            XPooled_sd = (sqrt(var(Features.Features)+0.01)+DataStats.Std)/2;   
        end                                                                     
%     end
    else                                    
%         XPooled_mean=DataStats.Mean;                                            
%         XPooled_sd=DataStats.Std;                                               
    end                                                                         
                                                                                
    Features.Mean=XPooled_mean;                                                 
    Features.Std=XPooled_sd;                                                    
                                                                                
    return                                                                      
end                                                                             
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X=Dataset.trainX;                                                             
if strcmp(optionsData.TrainTest,'Train') %either Train/Test data is processed.  
    X=Dataset.trainX; %data                                                     
    Features.Labels=Dataset.trainY;                                             
else                                                                            
    X=Dataset.testX; %data                                                      
    Features.Labels=Dataset.testY;                                   
end                                                                             
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    


if  strcmp(Domain,'Spatial')
    qq=SamplingFrom_Spatial(Dataset,X,options);
    
elseif strcmp(Domain,'Frequency')
    q1=SamplingFrom_FrequencyMagnitude(Dataset,X,options);
    q2=SamplingFrom_FrequencyPhase(Dataset,X,options);
    qq=[q1 q2];

elseif strcmp(Domain,'FrequencyPhase')
    qq=SamplingFrom_FrequencyPhase(Dataset,X,options);

elseif strcmp(Domain,'FrequencyMagnitude')
    qq=SamplingFrom_FrequencyMagnitude(Dataset,X,options);

elseif strcmp(Domain,'Hybrid_SM')
    q1=SamplingFrom_Spatial(Dataset,X,options);
    q2=SamplingFrom_FrequencyMagnitude(Dataset,X,options);
    qq=[q1 q2];

elseif strcmp(Domain,'Hybrid_SP')
    q1=SamplingFrom_Spatial(Dataset,X,options);
    q2=SamplingFrom_FrequencyPhase(Dataset,X,options);
    qq=[q1 q2];
    
elseif strcmp(Domain,'Hybrid_SMP')
    
    q1=SamplingFrom_Spatial(Dataset,X,options);
    q2=SamplingFrom_FrequencyMagnitude(Dataset,X,options);
    q3=SamplingFrom_FrequencyPhase(Dataset,X,options);
    
    qq=[q1 q2 q3];
    
end

if options{9}.SmoothingFlag
    BatchFile=strcat(savedir,int2str(options{1}.TotalNumberOfBatches),';',Dataset.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_[',int2str(options{9}.SmoothKernel),']_',num2str(options{9}.SD_SmoothingFilter),'_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
else
    BatchFile=strcat(savedir,int2str(options{1}.TotalNumberOfBatches),';',Dataset.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_[NO Smoothing]_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
end



if strcmp(optionsData.TrainTest,'Train')
    if isempty(DataStats)
        XPooled_mean = mean(qq);
        XPooled_sd = sqrt(var(qq)+0.01);
    else %for batch based processing, combine with previous batch stats
        XPooled_mean = (mean(qq)+DataStats.Mean)/2;
        XPooled_sd = (sqrt(var(qq)+0.01)+DataStats.Std)/2;
    end
else
    XPooled_mean=DataStats.Mean;
    XPooled_sd=DataStats.Std;
end

Features.Features=qq;
Features.Mean=XPooled_mean;
Features.Std=XPooled_sd;
Features.optionsAll=options;
Features.optionsPrev=options;
Features.RFData=ReceptiveFields;
Features.SparsifyFlag=optionsExtract.SparsifyFlag;
Features.Binarize=optionsNormalize.Binarize;
Features.SparsityMultiplier=optionsExtract.SparsityMultiplier;
Features.SummationType=optionsPool.SummationType;
Features.DatasetName=Dataset.DatasetName;
Features.UniqueLabels=Dataset.UniqueLabels;

if optionsData.Verbose
    fprintf('Feature Extraction done! :  %f\n',toc)
end

save(BatchFile,'Features','-v7.3');
end
