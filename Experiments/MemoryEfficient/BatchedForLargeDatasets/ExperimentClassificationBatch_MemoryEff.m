%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function performs experiments on specific computer vision dataset/task
%   step by step. LOADS features and TRAINS in batches,
%   specifically for large datasets.
%   MemoryEff refers to bundling of steps 4,5 and 6 for RAM
%   effieciency, inside one function named:
%   ExtractPoolNormalizeSingleLayerFeature_v2.
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

%put every option in a list, for monolithic feature computation in one
%function.
optionsAll{1}=optionsExtractFeatures;
optionsAll{2}=optionsPoolFeatures;
optionsAll{3}=optionsNormalizeBinarize;

%Load dataset and get important info
optionsData.BatchNumber=1;
Dataset=LoadDataset(DatasetName,RootPath,optionsData);

%Run unsupervised learning, or load precomputed data
load('F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\SavedData\KmeansRF_CIFAR10_NoRFs_200RFSize_6.mat')

for i=1:1:Dataset.NumberOfBatches
    %select a specific batch
    optionsData.BatchNumber=i;
    
    %load features of the batch
    LoadDatasetFeatures(DatasetName,RootPath,optionsData);
    NeuralNetFeatures_PooledNormalized_Train=NeuralNetFeatures;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %TODO: implement bagging, and feature subset selection as in Penn Treebank 
    %experiments
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %Supervised learning
    Model=SupervisedLearning(Model,NeuralNetFeatures_PooledNormalized_Train,Dataset.trainY,optionsSupervisedLearning);


end


%Measure performance on test data
%extract features for test data
optionsAll{1}.TrainOrTest='Test';
for i=1:1:Dataset.NumberOfBatches
    %select a specific batch
    optionsData.BatchNumber=i;
    
    %load features of the batch
    LoadDatasetFeatures(DatasetName,RootPath,optionsData);
    NeuralNetFeatures_PooledNormalized_Train=NeuralNetFeatures;
    
    %compute test error
    Performance=EvaluateSupervisedLearning(Model,NeuralNetFeatures_PooledNormalized_Test,Dataset.testY,optionsEvaluation);

end



