%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function loads the parameters and options for all the substages of
%the experiments on computer vision/machine learning. 
%Eg. ExperimentClassification.m
%
%
%From:
%TOU_ML
%Ozgur Yilmaz, Turgut Ozal University, Ankara
%Web: ozguryilmazresearch.net
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%Path for the root
RootPath='F:\Recurrent Holistic Vision v0.4\MatlabCode';
SavePath='F:\Recurrent Holistic Vision v0.4\SavedData';
ExperimentName='CIFAR10_SingleLayer'; %'CIFAR10_SingleLayer', 'PennTreeBank_Character_CA'
fprintf(strcat('EXPERIMENT: ', ExperimentName,'\n'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Dataset name
DatasetName='CIFAR10'; % 'CIFAR10', 'PennTreebank_Character', 'CIFAR10_YCbCr'
%LoadData options
optionsData.AutoLoad=true; %If epochs are used in training, always set to true
optionsData.Verbose=true;
optionsData.WhichData='Raw';       %'Raw', 'Features', 'ExpandedFeatures', 'Model'.
optionsData.DatasetName=DatasetName;
optionsData.TrainTest='Train'; %'Train', 'Test'
optionsData.Randomize=false;
optionsData.TrainDataRatio=1; %0.2;
optionsData.BatchNumber=0;
optionsData.TotalNumberOfBatches=1; %number of batches in datasets. ex/ for CIFAR10 it may be: 5, 10, or 50
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Kmeans parameters
NumberOfRFs=1600;
rfSize=6;
%Kmeans options
optionsKmeans.AutoLoad=true;
optionsKmeans.NumberOfRFs=NumberOfRFs;
optionsKmeans.rfSize=rfSize;
optionsKmeans.ImageIsGrayFlag=false;
optionsKmeans.GrayConvertFlag=false;
optionsKmeans.numPatches=10^6;
optionsKmeans.ShowCentroids=false;
optionsKmeans.SaveRFs=true;
optionsKmeans.SavePath=SavePath;
optionsKmeans.BinaryDescription = true; % yeni eklendi 24.01.2016
optionsKmeans.BitsPerRFs = 1024;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature extraction
% '2D', '3D' Extract features from 2D-space(from each channel seperately or
% 3D-space (RGB,YIQ,HSV,Gray is not suitable for 3D feature extraction)

optionsExtractFeatures.Perform=true;
optionsExtractFeatures.Type='ExtractPoolNormalizeSingleLayerFeature_v3';
%'ExtractBRIEF', 'ExtractPoolNormalizeSingleLayerFeature_v3'
if (optionsKmeans.BinaryDescription==1)
    optionsExtractFeatures.Type='ExtractPoolNormalizeBinaryFeature';
end
optionsExtractFeatures.SparsifyFlag=true;
optionsExtractFeatures.SparsityMultiplier=1;%0.94
optionsExtractFeatures.GPU=true;
optionsExtractFeatures.BatchSize=250; %100
optionsExtractFeatures.TrainOrTest=optionsData.TrainTest; %'Test'
optionsExtractFeatures.SaveFeatures=true;
optionsExtractFeatures.SavePath=SavePath;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






% This options set is meaningful only when we select 'optionsExtractFeatures.Type=ExtractBRIEF' above
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
optionsBriefFeatureExtraction.ColorType='RGB'; % 'RGB', 'Gray', 'HSV', 'YCbCr', 'YIQ';
optionsBriefFeatureExtraction.Type='BriefOverlappedPatches';
optionsBriefFeatureExtraction.Dimension='3D'; %'2D', '3D'; If you select 3D then be carefull to below parameters(give the high number for NumberOfTests(1))

if strcmp(optionsBriefFeatureExtraction.ColorType,'Gray')
    optionsBriefFeatureExtraction.Dimension='3D';
end
optionsBriefFeatureExtraction.SmoothingFlag=false; % 'true', 'false'
optionsBriefFeatureExtraction.SmoothKernel=[3 3]; %'[3 3]', [5 5], [9 9] %%valid for only BRIEF Extraction to smooth the image
optionsBriefFeatureExtraction.SD_SmoothingFilter=0.6; %0.1 means no smoothing will be done, use (SmoothingFlag=false) instead
%Number of Bits going to be created per channel [ch1 ch2 ch3]


% optionsBriefFeatureExtraction.NumberOfTests=[128 128 128];

%********************************************************
%___________Spatial Domain Parameters_____________:
optionsBriefFeatureExtraction.NumberOfTests1 =    [128 128 128];
optionsBriefFeatureExtraction.PatchSize1         =    [8 8]; % '[8 8]', Be careful while choosing these parameters
optionsBriefFeatureExtraction.Stride1              =    8;

%___________Frequency (Magnitude) Domain Parameters_______:
optionsBriefFeatureExtraction.NumberOfTests2 =    [70 70 70];
optionsBriefFeatureExtraction.PatchSize2         =    [4 4]; % '[8 8]', Be careful while choosing these parameters
optionsBriefFeatureExtraction.Stride2              =    4;

%___________Frequency (Phase) Domain Parameters__________:
optionsBriefFeatureExtraction.NumberOfTests3 =     [68 68 68];
optionsBriefFeatureExtraction.PatchSize3         =     [4 4]; % '[8 8]', Be careful while choosing these parameters
optionsBriefFeatureExtraction.Stride3              =    4;
%********************************************************

%Number of extracted bits per patches, 
%If you chose 3D feature extraction , then please type the same numbers for
%all channels. ([1024 1024 1024]]); the first one is assumed in
%computations!!!
%make sense that it is idle to choose both 3D and Gray Color Space!
optionsBriefFeatureExtraction.Domain='Spatial'; % 
% 1 - 'Spatial' - All samples are from spatial space
% 2 - 'Frequency'               - Samples from frequency domain (Both from magnitude and Phase components)
% 3 - 'FrequencyPhase'        - Samples from phase components of frequency domain
% 4 - 'FrequencyMagnitude' - Samples from magnitude components of frequency domain 
% 5 - 'Hybrid_SM'               - Samples from Hybrid domain (Spatial & magnitude_frequency)
% 6 - 'Hybrid_SP'                - Samples from Hybrid domain (Spatial & phase_frequency)
% 7 - 'Hybrid_SMP'              - Samples from Hybrid domain (Spatial & magnitude_frequency & phase_frequency);


if (optionsKmeans.BinaryDescription==1)
    if strcmp(optionsBriefFeatureExtraction.Domain,'Spatial')
        optionsBriefFeatureExtraction.PatchSize1         =    [rfSize rfSize]; 
        if      strcmp(optionsBriefFeatureExtraction.Dimension,'3D')
            optionsBriefFeatureExtraction.NumberOfTests1 =    [optionsKmeans.BitsPerRFs optionsKmeans.BitsPerRFs optionsKmeans.BitsPerRFs];
            optionsBriefFeatureExtraction.Stride1 = 0;
        elseif strcmp(optionsBriefFeatureExtraction.Dimension,'2D')
            NoOfBits_ch1 = floor(optionsKmeans.BitsPerRFs/3);
            NoOfBits_ch2 = floor(optionsKmeans.BitsPerRFs/3);
            NoOfBits_ch3 = optionsKmeans.BitsPerRFs - NoOfBits_ch1-NoOfBits_ch2;
            optionsBriefFeatureExtraction.NumberOfTests1 =    [NoOfBits_ch1 NoOfBits_ch2 NoOfBits_ch3];
        end
        
        
        
%     elseif strcmp(optionsBriefFeatureExtraction.Domain,'Frequency')
%         optionsBriefFeatureExtraction.NumberOfTests2 =    [floor(optionsKmeans.BitsPerRFs*0.5) floor(options)];
%         optionsBriefFeatureExtraction.PatchSize2         =    [4 4];
%         optionsBriefFeatureExtraction.Stride2              =    4;
%         
%         optionsBriefFeatureExtraction.NumberOfTests3 =     [68 68 68];
%         optionsBriefFeatureExtraction.PatchSize3         =     [4 4];
%         optionsBriefFeatureExtraction.Stride3              =    4;
%         
%         
%     elseif strcmp(optionsBriefFeatureExtraction.Domain,'FrequencyPhase')
%     elseif strcmp(optionsBriefFeatureExtraction.Domain,'FrequencyMagnitude')
%     elseif strcmp(optionsBriefFeatureExtraction.Domain,'Hybrid_SM')
%     elseif strcmp(optionsBriefFeatureExtraction.Domain,'Hybrid_SP')
%     elseif strcmp(optionsBriefFeatureExtraction.Domain,'Hybrid_SMP')
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Pooling and post processing
optionsPoolFeatures.SummationType='Coarse'; %'Coarse' 'Fine', 'Coarse', 'Single', or 'None' (Not recommended!)
optionsPoolFeatures.IntegralImage=false; %TODO Implement this.
optionsPoolFeatures.GPU=false;
optionsPoolFeatures.SaveFeatures=false;
optionsPoolFeatures.SavePath=SavePath;

optionsNormalizeBinarize.Binarize=false; %true
optionsNormalizeBinarize.BinarizeType='QuantizeIntegerBinary'; %'Simple', 'QuantizeIndicatorBinary', 'QuantizeInteger', 'QuantizeIntegerBinary' or 'KMeans'
optionsNormalizeBinarize.BinarizationThreshold=0; %For 'Simple' BinarizeType  %1 corresponds to 15% ones, -1 to 88% ones, -0.6 to 74% ones.
%32 bin: [-2.5,-2.25,-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,-0.15,-0.05,0,0.05,0.15,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4,4.5,5]
%16 bin: [-2,-1.5,-1,-0.75,-0.5,-0.25,-0.15,0.15,0.25,0.5,0.75,1,1.5,2,3]
%8 bin:  [-2, -1, -0.5, 0, 0.5, 1, 2];
%4 bin:  [-0.5, 0, 0.5]
optionsNormalizeBinarize.QuantizationThresholds= [-2,-1.5,-1,-0.75,-0.5,-0.25,-0.15,0.15,0.25,0.5,0.75,1,1.5,2,3]; %these bins are used for 'QuantizeBinary' or  'QuantizeInteger' BinarizeType
optionsNormalizeBinarize.GPU=false;
optionsNormalizeBinarize.SaveFeatures=true; %WARNING: If Epoch based training is used, always set to true.
optionsNormalizeBinarize.SavePath=SavePath;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Expansion with other algorithms
optionsExpansion.Perform=false;
optionsExpansion.Type='CA'; %'CA' 
optionsExpansion.CombinationOperation='NormalizedSummation'; %'XOR', 'NormalizedSummation'
optionsExpansion.ResetSequence=true;
optionsExpansion.Treset=100;
optionsExpansion.CARule=110;
optionsExpansion.CAIter=9;
optionsExpansion.CAPerm=16;
optionsExpansion.SamplingRatio=0;%1*1/(optionsExpansion.CAIter*optionsExpansion.CAPerm);
optionsExpansion.Normalize=false;
optionsExpansion.ConvertInteger=false;
optionsExpansion.IntegerBitSize=8;
optionsExpansion.GPU=true;
optionsExpansion.BatchSize=100;%100
optionsExpansion.SaveFeatures=true; %WARNING: If Epoch based training is used, always set to true.
optionsExpansion.SavePath=SavePath;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Supervised learning
%PLEASE NOTE THAT: Supervised learning subsection offers a selection
%of basic algorithmic combinations. BUT, not all combinations make sense.
%Think hard before trying many things together.
optionsSupervisedLearning.Task='Classification'; %Classification or Regression
optionsSupervisedLearning.ModelType='Softmax'; %'Softmax', 'SVM', 'KernelSVM', 'FLANN', 'SVM2'
% FLANN - Fast Library for Approximate Nearest Neighbors
optionsSupervisedLearning.Epochs=1; %WARNING: Should be set to 1, if Ensemble learning is used.
optionsSupervisedLearning.EpochCrossValidation=(true && optionsSupervisedLearning.Epochs>1); %not true if there is only one epoch
optionsSupervisedLearning.EpochCrossValidationThreshold=-100; %NOTE: make -100 if no early stopping is used, i.e. EpochIterationDiagnostics=true;
optionsSupervisedLearning.EpochTrainSplitRatio=1; %0.9
optionsSupervisedLearning.Greedy=true;
optionsSupervisedLearning.EpochIterationDiagnostics=false; %Either use EpochIterationDiagnostics, or use EpochCrossValidation. Choose...
optionsSupervisedLearning.FeatureSubsetRatio=1; %0.2
optionsSupervisedLearning.lambda=1;%0.0000000001;%1e-3;%1e-1
optionsSupervisedLearning.rbfScale=100;
optionsSupervisedLearning.SV_Ratio=1;
optionsSupervisedLearning.GridSearch=true; %implemented for ModelType='Softmax' and 'KernelSVM'. WARNING: if Epochs is larger than 1, then grid search is very wasteful.
optionsSupervisedLearning.GridSearch_Fine=true;
optionsSupervisedLearning.GridSearchDepth=5; %optimized for 5
optionsSupervisedLearning.GridSearchDataRatio=1; %1 for a propoer cross validation data partition (0.8 partition is already hard coded)
optionsSupervisedLearning.EarlyStopping=false; %implemented for ModelType='Softmax'
optionsSupervisedLearning.EarlyStoppingMaxIter=3; %50. 
optionsSupervisedLearning.EarlyStoppingMaxFunEvals=2*optionsSupervisedLearning.EarlyStoppingMaxIter; 
optionsSupervisedLearning.EarlyStoppingImpThreshold=-0.5;%0 %0.05
optionsSupervisedLearning.EarlyStoppingFailureAllowed=2;
optionsSupervisedLearning.IterationDiagnostics=false;
optionsSupervisedLearning.MaxIter=1000;%3 for epoch based, 100 for ensemble
optionsSupervisedLearning.MaxFunEvals=2*optionsSupervisedLearning.MaxIter;
optionsSupervisedLearning.SVM_C=100;
optionsSupervisedLearning.GPU=true; %not recommended for ModelType='SVM'
optionsSupervisedLearning.SaveModel=true; %false is used for memory efficiency
optionsSupervisedLearning.SavePath=SavePath;
%A little WARNING. Saved model files are not distinguisable from each
%other, depending on parameters. Thus if you change the supervised learning
%parameters, and run again it will override the .mat file. This is not the
%case for features file (almost!).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
optionsFLANN.TrainsetBatcheNumber=50; %Divide the train set while searching NN into this number of batches. If (=1) then 
% it uses the entire train set to search the NN of current test point
optionsFLANN.k=13;
optionsFLANN.build_params_algorithm='autotuned';
optionsFLANN.build_params_target_precision=0.90;
optionsFLANN.build_params_build_weight=0.01;
optionsFLANN.build_memory_weight=0;
optionsFLANN.build_sample_fraction=0.2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Task/performance evaluation
optionsEvaluation.Task=optionsSupervisedLearning.Task;
optionsEvaluation.ModelType=optionsSupervisedLearning.ModelType;
optionsEvaluation.Ensemble=(false || optionsSupervisedLearning.FeatureSubsetRatio<1) && optionsSupervisedLearning.Epochs==1; 
%WARNING: Ensemble should be true, if a subset of the features are selected during
%training for each batch. It should be false if Epoch>1.
optionsEvaluation.EnsembleModelIndependence=true; %if this is true, then the initial condition for each model is random. If not, an ensemble model learning uses warm start
optionsEvaluation.EnsembleDecisionType='Majority'; %'Mean', 'Median', 'Majority'
optionsEvaluation.GPU=false;
optionsEvaluation.SavePerformance=true;
optionsEvaluation.SavePath=SavePath;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%















%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Options for saving the Results of Experiment
% Fixes the feature fileNames

Domain=optionsBriefFeatureExtraction.Domain;
Dimension=optionsBriefFeatureExtraction.Dimension;
NumberOfTests1=optionsBriefFeatureExtraction.NumberOfTests1; PatchSize1=optionsBriefFeatureExtraction.PatchSize1; Stride1=optionsBriefFeatureExtraction.Stride1;
NumberOfTests2=optionsBriefFeatureExtraction.NumberOfTests2; PatchSize2=optionsBriefFeatureExtraction.PatchSize2; Stride2=optionsBriefFeatureExtraction.Stride2;
NumberOfTests3=optionsBriefFeatureExtraction.NumberOfTests3; PatchSize3=optionsBriefFeatureExtraction.PatchSize3; Stride3=optionsBriefFeatureExtraction.Stride3;

if strcmp(Domain,'Spatial')
    if strcmp(Dimension,'3D')
        SampleNo=strcat('(S%[',int2str(PatchSize1),'];',int2str(Stride1),';{',int2str(NumberOfTests1(1)),'}%)');
    elseif strcmp(Dimension,'2D')
        SampleNo=strcat('(S%[',int2str(PatchSize1),'];',int2str(Stride1),';{',int2str(NumberOfTests1),'}%)');    
    end
elseif strcmp(Domain,'Frequency')
     if strcmp(Dimension,'3D')
         SampleNo=strcat('(M%[',int2str(PatchSize2),'];',int2str(Stride2),';{',int2str(NumberOfTests2(1)),'}% & P%[',int2str(PatchSize3),'];',int2str(Stride3),';{',int2str(NumberOfTests3(1)),'}%)');
    elseif strcmp(Dimension,'2D')
        SampleNo=strcat('(M%[',int2str(PatchSize2),'];',int2str(Stride2),';{',int2str(NumberOfTests2),'}% & P%[',int2str(PatchSize3),'];',int2str(Stride3),';{',int2str(NumberOfTests3),'}%)');
    end
elseif strcmp(Domain,'FrequencyPhase')
     if strcmp(Dimension,'3D')
        SampleNo=strcat('(P%[',int2str(PatchSize3),'];',int2str(Stride3),';{',int2str(NumberOfTests3(1)),'}%)');
    elseif strcmp(Dimension,'2D')
        SampleNo=strcat('(P%[',int2str(PatchSize3),'];',int2str(Stride3),';{',int2str(NumberOfTests3),'}%)');
    end
elseif strcmp(Domain,'FrequencyMagnitude')
      if strcmp(Dimension,'3D')
        SampleNo=strcat('(M%[',int2str(PatchSize2),'];',int2str(Stride2),';{',int2str(NumberOfTests2(1)),'}%)');
    elseif strcmp(Dimension,'2D')
        SampleNo=strcat('(M%[',int2str(PatchSize2),'];',int2str(Stride2),';{',int2str(NumberOfTests2),'}%)');
     end
elseif strcmp(Domain,'Hybrid_SM')
     if strcmp(Dimension,'3D')
            SampleNo=strcat('(S%[',int2str(PatchSize1),'];',int2str(Stride1),';{',int2str(NumberOfTests1(1)),'}% & M%[',int2str(PatchSize2),'];',int2str(Stride2),';{',int2str(NumberOfTests2(1)),'}%)');
    elseif strcmp(Dimension,'2D')
            SampleNo=strcat('(S%[',int2str(PatchSize1),'];',int2str(Stride1),';{',int2str(NumberOfTests1),'}% & M%[',int2str(PatchSize2),'];',int2str(Stride2),';{',int2str(NumberOfTests2),'}%)');
    end
elseif strcmp(Domain,'Hybrid_SP')
     if strcmp(Dimension,'3D')
            SampleNo=strcat('(S%[',int2str(PatchSize1),'];',int2str(Stride1),';{',int2str(NumberOfTests1(1)),'}% & P%[',int2str(PatchSize3),'];',int2str(Stride3),';{',int2str(NumberOfTests3(1)),'}%)');
    elseif strcmp(Dimension,'2D')
            SampleNo=strcat('(S%[',int2str(PatchSize1),'];',int2str(Stride1),';{',int2str(NumberOfTests1),'}% & P%',int2str(PatchSize3),'];',int2str(Stride3),';{',int2str(NumberOfTests3),'}%)');
    end
elseif strcmp(Domain,'Hybrid_SMP')
    if strcmp(Dimension,'3D')
            SampleNo=strcat('(S%[',int2str(PatchSize1),'];',int2str(Stride1),';{',int2str(NumberOfTests1(1)),'}% & M%[',int2str(PatchSize2),'];',int2str(Stride2),';{',int2str(NumberOfTests2(1)),'}% & P%[',int2str(PatchSize3),'];',int2str(Stride3),';{',int2str(NumberOfTests3(1)),'}%)');
    elseif strcmp(Dimension,'2D')
            SampleNo=strcat('(S%[',int2str(PatchSize1),'];',int2str(Stride1),';{',int2str(NumberOfTests1),'}% & M%[',int2str(PatchSize2),'];',int2str(Stride2),';{',int2str(NumberOfTests2),'}% & P%[',int2str(PatchSize3),'];',int2str(Stride3),';{',int2str(NumberOfTests3),'}%)');
    end
end
optionsBriefFeatureExtraction.SampleNo=SampleNo;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% savedir=strcat(RootPath,'\BRIEF\BriefOverlappedPatches\Saved BRIEF_OverlappedPatches Features\',options{9}.Domain,SampleNo,TrainTest,'-');
% if options{9}.SmoothingFlag
%     BatchFile=strcat(savedir,int2str(optionsData.TotalNumberOfBatches),';',DatasetName,'_',optionsBriefFeatureExtraction.Dimension,'_',optionsBriefFeatureExtraction.ColorType,'_[',int2str(optionsBriefFeatureExtraction.SmoothKernel),']_',num2str(optionsBriefFeatureExtraction.SD_SmoothingFilter),'_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
% else
%     BatchFile=strcat(savedir,int2str(optionsData.TotalNumberOfBatches),';',DatasetName,'_',optionsBriefFeatureExtraction.Dimension,'_',optionsBriefFeatureExtraction.ColorType,'_[NO Smoothing]_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
% end
% 



