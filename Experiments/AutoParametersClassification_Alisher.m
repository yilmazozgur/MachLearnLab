function [NumberOfRFs,rfSize,DatasetName,optionsAll]=AutoParametersClassification_Alisher(Epoch,lamda)
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


fprintf('It is running correctly!!!\n')
%Path for the root
RootPath='F:\Recurrent Holistic Vision v0.4\MatlabCode';
SavePath='F:\Recurrent Holistic Vision v0.4\SavedData';
ExperimentName='CIFAR10_SingleLayer'; %'CIFAR10_SingleLayer', 'PennTreeBank_Character_CA'
fprintf(strcat('EXPERIMENT: ', ExperimentName,'\n'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Dataset name
DatasetName='CIFAR10'; %'CIFAR10', 'PennTreebank_Character'
%LoadData options
optionsData.AutoLoad=true; %If epochs are used in training, always set to true
optionsData.Verbose=true;
optionsData.WhichData='Model'; %'Raw', 'Features', 'ExpandedFeatures', 'Model'.
optionsData.DatasetName=DatasetName;
optionsData.TrainTest='Train'; %'Train', 'Test'
optionsData.Randomize=false;
optionsData.TrainDataRatio=1; %0.2;
optionsData.BatchNumber=1;
optionsData.TotalNumberOfBatches=100; %number of batches in datasets. ex/ for CIFAR10 it may be: 5, 10, or 50
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Kmeans parameters
NumberOfRFs=50;
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature extraction
% '2D', '3D' Extract features from 2D-space(from each channel seperately or
% 3D-space (RGB,YIQ,HSV,Gray is not suitable for 3D feature extraction)

optionsExtractFeatures.Perform=true;
optionsExtractFeatures.Type='ExtractBRIEF';
%'ExtractBRIEF', 'ExtractPoolNormalizeSingleLayerFeature_v3'
optionsExtractFeatures.SparsifyFlag=false;
optionsExtractFeatures.SparsityMultiplier=1;%0.94
optionsExtractFeatures.GPU=true;
optionsExtractFeatures.BatchSize=100;
optionsExtractFeatures.TrainOrTest=optionsData.TrainTest; %'Test'
optionsExtractFeatures.SaveFeatures=true;
optionsExtractFeatures.SavePath=SavePath;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






% This options set is meaningful only when we select 'optionsExtractFeatures.Type=ExtractBRIEF' above
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
optionsBriefFeatureExtraction.ColorType='RGB'; % 'RGB', 'Gray', 'HSV', 'YCbCr', 'YIQ';
optionsBriefFeatureExtraction.Type='BriefOverlappedPatches'; % 'BriefSubImages', 'BriefOverlappedPatches'
optionsBriefFeatureExtraction.Dimension='3D'; %'2D', '3D'; If you select 3D then be carefull to below parameters(give the high number for NumberOfTests(1))
optionsBriefFeatureExtraction.SmoothingFlag=false; % 'true', 'false'
optionsBriefFeatureExtraction.SmoothKernel=[3 3]; %'[3 3]', [5 5], [9 9] %%valid for only BRIEF Extraction to smooth the image
optionsBriefFeatureExtraction.SD_SmoothingFilter=0.4; %0.1 means no smoothing will be done, use (SmoothingFlag=false) instead
%Number of Bits going to be created per channel [ch1 ch2 ch3]
optionsBriefFeatureExtraction.NumberOfTests=[8192 8192 8192]; %Number of extracted bits per patches, 
%If you chose 3D feature extraction , then please type the same numbers for
%all channels. ([1024 1024 1024]]); the first one is assumed in
%computations!!!
%make sense that it is idle to choose both 3D and Gray Color Space!


%Parameteres for ExtarctBRIEF_OverlappedPatches
optionsBriefFeatureExtraction.PatchSize=[14 14]; % '[8 8]', Be careful while choosing these parameters
optionsBriefFeatureExtraction.Stride=2; %step-size: '2', '3', '4' etc..
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Pooling and post processing
optionsPoolFeatures.SummationType='Coarse'; %'Coarse' 'Fine', 'Coarse', 'Single', or 'None' (Not recommended!)
optionsPoolFeatures.IntegralImage=false; %TODO Implement this.
optionsPoolFeatures.GPU=false;
optionsPoolFeatures.SaveFeatures=false;
optionsPoolFeatures.SavePath=SavePath;

optionsNormalizeBinarize.Binarize=true; %true
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
optionsSupervisedLearning.ModelType='Softmax'; %'Softmax', 'SVM' or 'KernelSVM'
optionsSupervisedLearning.Epochs=Epoch; %WARNING: Should be set to 1, if Ensemble learning is used.
optionsSupervisedLearning.EpochCrossValidation=(true && optionsSupervisedLearning.Epochs>1); %not true if there is only one epoch
optionsSupervisedLearning.EpochCrossValidationThreshold=-100; %NOTE: make -100 if no early stopping is used, i.e. EpochIterationDiagnostics=true;
optionsSupervisedLearning.EpochTrainSplitRatio=0.9;
optionsSupervisedLearning.Greedy=true;
optionsSupervisedLearning.EpochIterationDiagnostics=true; %Either use EpochIterationDiagnostics, or use EpochCrossValidation. Choose...
optionsSupervisedLearning.FeatureSubsetRatio=1; %0.2
optionsSupervisedLearning.lambda=lamda;%1e-3;%1e-1
optionsSupervisedLearning.rbfScale=100;
optionsSupervisedLearning.SV_Ratio=1;
optionsSupervisedLearning.GridSearch=false; %implemented for ModelType='Softmax' and 'KernelSVM'. WARNING: if Epochs is larger than 1, then grid search is very wasteful.
optionsSupervisedLearning.GridSearch_Fine=true;
optionsSupervisedLearning.GridSearchDepth=5; %optimized for 5
optionsSupervisedLearning.GridSearchDataRatio=1; %1 for a propoer cross validation data partition (0.8 partition is already hard coded)
optionsSupervisedLearning.EarlyStopping=false; %implemented for ModelType='Softmax'
optionsSupervisedLearning.EarlyStoppingMaxIter=3; %50. 
optionsSupervisedLearning.EarlyStoppingMaxFunEvals=2*optionsSupervisedLearning.EarlyStoppingMaxIter; 
optionsSupervisedLearning.EarlyStoppingImpThreshold=-0.5;%0 %0.05
optionsSupervisedLearning.EarlyStoppingFailureAllowed=2;
optionsSupervisedLearning.IterationDiagnostics=false;
optionsSupervisedLearning.MaxIter=3;%3 for epoch based, 100 for ensemble
optionsSupervisedLearning.MaxFunEvals=2*optionsSupervisedLearning.MaxIter;
optionsSupervisedLearning.GPU=true; %not recommended for ModelType='SVM'
optionsSupervisedLearning.SaveModel=true; %false is used for memory efficiency
optionsSupervisedLearning.SavePath=SavePath;
%A little WARNING. Saved model files are not distinguisable from each
%other, depending on parameters. Thus if you change the supervised learning
%parameters, and run again it will override the .mat file. This is not the
%case for features file (almost!).
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


optionsAll{1}=optionsData;
optionsAll{2}=optionsExtractFeatures;
optionsAll{3}=optionsPoolFeatures;
optionsAll{4}=optionsNormalizeBinarize;
optionsAll{5}=optionsExpansion;
optionsAll{6}=optionsSupervisedLearning;
optionsAll{7}=optionsEvaluation;
optionsAll{8}=optionsKmeans;
optionsAll{9}=optionsBriefFeatureExtraction;
optionsAll{10}=RootPath;


end