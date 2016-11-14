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
RootPath='F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\MatlabCode';
SavePath='F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\SavedData';
ExperimentName='PennTreeBank_Character ESN'; %'CIFAR10_SingleLayer', 'PennTreeBank_Character CA', 'TwentyBit CA'
fprintf(strcat('EXPERIMENT: ', ExperimentName,'\n'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Dataset name
DatasetName='TwentyBit'; %'CIFAR10', 'PennTreebank_Character', 'TwentyBit'
%LoadData options
optionsData.AutoLoad=true; %If epochs are used in training, always set to true
optionsData.Verbose=true;
optionsData.WhichData='Raw'; %'Raw', 'Features', 'ExpandedFeatures', 'Model'.
optionsData.DatasetName=DatasetName;
optionsData.TrainTest='Train'; %'Train', 'Test'
optionsData.Randomize=false;
optionsData.TrainDataRatio=1; %0.2;
optionsData.BatchNumber=0;
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
optionsExtractFeatures.Perform=false;
optionsExtractFeatures.SparsifyFlag=true;
optionsExtractFeatures.SparsityMultiplier=1;%0.94
optionsExtractFeatures.GPU=true;
optionsExtractFeatures.BatchSize=100;
optionsExtractFeatures.TrainOrTest=optionsData.TrainTest; %'Test'
optionsExtractFeatures.SaveFeatures=false;
optionsExtractFeatures.SavePath=SavePath;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Pooling and post processing
optionsPoolFeatures.SummationType='Coarse'; %'Finest' 'Fine', 'Coarse', 'Single', or 'None' (Not recommended!)
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
optionsExpansion.Perform=true;
optionsExpansion.Type='ESN'; %'CA' , 'ESN' 
optionsExpansion.CAContinuous=false;
optionsExpansion.MappingOperation='RandomPermutation'; %'RandomPermutation', 'LinearProjection'
optionsExpansion.LinearProjectionThreshold=0;%0.25;
optionsExpansion.CombinationOperation='NormalizedSummation'; %'XOR', 'NormalizedSummation', 'ESNType'
optionsExpansion.ContinuousCASumTerm=0.25;%0.325;
optionsExpansion.ContinuousCAMultTerm=1;
optionsExpansion.ContinuousCAWeightNewSeq=0.5;
optionsExpansion.ResetSequence=true;
optionsExpansion.Treset=30; %(T+2*M)/BundlePeriod for 20BitTask
optionsExpansion.WarmUpSequenceRatio=0;
optionsExpansion.CARule=150;
optionsExpansion.CAIter=2;
optionsExpansion.CAPerm=200;%200
optionsExpansion.SamplingRatio=0;%1*1/(optionsExpansion.CAIter*optionsExpansion.CAPerm); %WARNING: does not work with sequence tasks
optionsExpansion.Normalize=false;
optionsExpansion.ConvertInteger=false;
optionsExpansion.IntegerBitSize=8;
%ESN parameters
optionsExpansion.nInternalUnits = 1000; % Size of reservoir
optionsExpansion.connectivity = 0.005; %0.005; % average connectivity of reservoir (1 = fully connected)
optionsExpansion.inScaling = 0.000001; %0.000001; %([0.00001*[1 1 1 1 1], 0.000001, 1])'; % how the input weights are scaled
optionsExpansion.SR = 1; % Weight matrix spectral radius
optionsExpansion.LR = 1; % smoothing rate, from (0,1]. Low values = much smoothing.
optionsExpansion.W =  []; %SR * generate_internal_weights(nInternalUnits, connectivity);
optionsExpansion.Win = []; %2 * (rand(nInternalUnits, 7) - 0.5) * diag(inScaling);
optionsExpansion.GPU=true;
optionsExpansion.BatchSize=100;
optionsExpansion.SaveFeatures=false; %WARNING: If Epoch based training is used, always set to true.
optionsExpansion.SavePath=SavePath;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Supervised learning
%PLEASE NOTE THAT: Supervised learning subsection offers a selection
%of basic algorithmic combinations. BUT, not all combinations make sense.
%Think hard before trying many things together.
optionsSupervisedLearning.Task='Regression'; %Classification or Regression WARNING: ESN expansion seem to work only with Regression, and CA favors Classification!!!
optionsSupervisedLearning.ModelType='Softmax'; %'Softmax', 'SVM' or 'KernelSVM'
optionsSupervisedLearning.Epochs=1; %WARNING: Should be set to 1, if Ensemble learning is used.
optionsSupervisedLearning.EpochCrossValidation=(false && optionsSupervisedLearning.Epochs>1); %not true if there is only one epoch
optionsSupervisedLearning.EpochCrossValidationThreshold=0;
optionsSupervisedLearning.EpochTrainSplitRatio=0.9;
optionsSupervisedLearning.EpochIterationDiagnostics=false; %Either use EpochIterationDiagnostics, or use EpochCrossValidation. Choose...
optionsSupervisedLearning.FeatureSubsetRatio=1; %0.2
optionsSupervisedLearning.lambda=1e-3;%1e-1
optionsSupervisedLearning.rbfScale=100;
optionsSupervisedLearning.SV_Ratio=1;
optionsSupervisedLearning.GridSearch=false; %implemented for ModelType='Softmax' and 'KernelSVM'. WARNING: if Epochs is larger than 1, then grid search is very wasteful.
optionsSupervisedLearning.GridSearch_Fine=true;
optionsSupervisedLearning.GridSearchDepth=5; %optimized for 5
optionsSupervisedLearning.GridSearchDataRatio=1; %1 for a propoer cross validation data partition (0.8 partition is already hard coded)
optionsSupervisedLearning.EarlyStopping=false; %implemented for ModelType='Softmax'
optionsSupervisedLearning.EarlyStoppingMaxIter=10; %50. 
optionsSupervisedLearning.EarlyStoppingMaxFunEvals=2*optionsSupervisedLearning.EarlyStoppingMaxIter; 
optionsSupervisedLearning.EarlyStoppingImpThreshold=0; %0.05
optionsSupervisedLearning.EarlyStoppingFailureAllowed=3;
optionsSupervisedLearning.IterationDiagnostics=false;
optionsSupervisedLearning.MaxIter=100;%3 for epoch based, 100 for ensemble
optionsSupervisedLearning.MaxFunEvals=2*optionsSupervisedLearning.MaxIter;
optionsSupervisedLearning.GPU=true; %not recommended for ModelType='SVM'
optionsSupervisedLearning.SaveModel=true;
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


