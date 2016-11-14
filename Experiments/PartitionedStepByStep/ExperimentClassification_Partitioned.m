%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function performs experiments on specific computer vision dataset/task
%   step by step
%
%%% Workflow:
% 1. Parameters are loaded in a separate function
% 2. Dataset is loaded using LoadDataset function
% 3. Unsupervised learning is done on data, optional (eg. KmeansUnsupervised)
% 4. Features are extracted or neural network is applied on the dataset
% 5. Extracted features are pooled (or other reduction techniques)
% 6. Features are expanded by other algorithms (eg. Cellular Automata Reservoir)    
% 7. Supervised learing is performed using minFunc
% 8. Performance metrics are calculated, depending on the dataset/task.
%
%From:
%TOU_ML
%Ozgur Yilmaz, Turgut Ozal University, Ankara
%Web: ozguryilmazresearch.net
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Load parameters and options for this experiment
ParametersClassification

%Load dataset
Dataset=LoadDataset(DatasetName,RootPath,optionsData);

%Run unsupervised learning
% ReceptiveFields=KmeansUnsupervised(Dataset,NumberOfRFs,rfSize,optionsKmeans);
load('F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\SavedData\KmeansRF_CIFAR10_NoRFs_200RFSize_6.mat')

%Extract single layer neural net features
NeuralNetFeatures=ExtractSingleLayerFeature_v2(Dataset,ReceptiveFields,optionsExtractFeatures);

%Pool the features and PostProcess (normalize and binarize data)
PooledFeatures=PoolFeatures(NeuralNetFeatures,optionsPoolFeatures);
clear NeuralNetFeatures %flush memory for unused
NormalizedFeaturesTrain=NormalizeBinarize(PooledFeatures,[],optionsNormalizeBinarize);
clear PooledFeatures %flush memory for unused

%Expand the space with cellular automata features, or polynomial/kernel
%features etc. see minFunc_examples.m
%Coming soon...

%Supervised learning
Model=SupervisedLearning([],NormalizedFeaturesTrain,Dataset.trainY,optionsSupervisedLearning);

%Measure performance on test data
%extract features for test data
optionsExtractFeatures.TrainOrTest='Test';
NeuralNetFeatures=ExtractSingleLayerFeature_v2(Dataset,ReceptiveFields,optionsExtractFeatures);
PooledFeatures=PoolFeatures(NeuralNetFeatures,optionsPoolFeatures);
%give the training data stats for normalization
DataStats.Mean=NormalizedFeaturesTrain.Mean;
DataStats.Std=NormalizedFeaturesTrain.Std;
NormalizedFeaturesTest=NormalizeBinarize(PooledFeatures,DataStats,optionsNormalizeBinarize);

%compute test error
Performance=EvaluateSupervisedLearning(Model,NormalizedFeaturesTest,Dataset.testY,optionsEvaluation);



