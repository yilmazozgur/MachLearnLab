%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function performs precomputation of features on a large dataset,
%   batch by batch. And saves the features in separate file for each batch.
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

%load an initial batch for getting info on dataset, i.e. NumberOfBatches
optionsData.BatchNumber=1;
Dataset=LoadDataset(DatasetName,RootPath,optionsData);

%Run unsupervised learning, for first batch only.
ReceptiveFields=KmeansUnsupervised(Dataset,NumberOfRFs,rfSize,optionsKmeans);
% load('F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\SavedData\KmeansRF_CIFAR10_NoRFs_200RFSize_6.mat')


for i=1:1:Dataset.NumberOfBatches
    %select a specific batch
    optionsData.BatchNumber=i;
    
    %Load dataset
    Dataset=LoadDataset(DatasetName,RootPath,optionsData);

    %Extract single layer neural net features
    NeuralNetFeatures_PooledNormalized_Train=ExtractPoolNormalizeSingleLayerFeature_v2(Dataset,ReceptiveFields,[],optionsAll);
    MeanIter(i,:)=NeuralNetFeatures_PooledNormalized_Train.Mean;
    StdIter(i,:,:)=NeuralNetFeatures_PooledNormalized_Train.Std;
    
    %Expand the space with cellular automata features, or polynomial/kernel
    %features etc. see minFunc_examples.m
    %Coming soon...

    %TODO:
    %save
end

%extract features for test data
optionsAll{1}.TrainOrTest='Test';
DataStats.Mean=mean(MeanIter);
DataStats.Std=mean(StdIter);

for i=1:1:Dataset.NumberOfBatches
    %select a specific batch
    optionsData.BatchNumber=i;
    
    %Load dataset
    Dataset=LoadDataset(DatasetName,RootPath,optionsData);

    %Extract single layer neural net features
    NeuralNetFeatures_PooledNormalized_Test=ExtractPoolNormalizeSingleLayerFeature_v2(Dataset,ReceptiveFields,DataStats,optionsAll);
   
    %Expand the space with cellular automata features, or polynomial/kernel
    %features etc. see minFunc_examples.m
    %Coming soon...

    %TODO:
    %save
end



