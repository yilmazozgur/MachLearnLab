%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function normalizes and binarizes features of a dataset,
% for vision/machine learning tasks.
%
%%% Input:
%PooledFeatures : A struct (possibly created by PoolFeatures.m) that holds
%   a set of features for many data instances. 
%DataStats: Mean and Std of features that is going to be used for
%   normalization. (Subtract mean from the feature and divide by std). It 
%   It is given only for Test Data features. For Train Data features, it is 
%   computed here.     
%options: struct for options. 
%options.Binarize: a flag that determines if the output feature vector will
%   be binary.
%optionsNormalize.BinarizationThreshold: After mean and std normalization, 
%   this threshold determines which entries are non-zero in binary vector. Default=0;
%options.GPU: enable GPU computing. Default=false.
%options.SaveFeatures: Save the results to the Root when finished.
%   Default=false.
%options.SavePath: the path to save the results if SaveRFs flag is up.
%   Default='';
%
%%% Output:
%NeuralNetFeatures: A struct that holds neural network responses of size
%   ImageDim1*ImageDim1*numRFs for each image.
%
%
%From:
%TOU_ML
%Ozgur Yilmaz, Turgut Ozal University, Ankara
%Web: ozguryilmazresearch.net
%Ref: It is based on Yilmaz et al. 2015.
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function NormalizedFeatures=NormalizeBinarize(PooledFeatures,DataStats,options)

%measure computation time
tic

%default options, if not given by the user
if nargin < 3
    options.Binarize=true;
    options.BinarizationThreshold=0;
    options.GPU=false;
    options.SaveFeatures=false;
    options.SavePath='F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\SavedData';
end

%extract the raw data and image dimensions
ReceptiveFields=PooledFeatures.RFData;

%infer knowledge on the RFs
numCentroids = size(ReceptiveFields.RFs,1);
rfSize=ReceptiveFields.RFSize;

%Show what is to be done
fprintf('Normalize/Binarize Features Start: \n'); 

NNFeature=zeros(size(PooledFeatures.Features,2),size(PooledFeatures.Features{1},2));
%form a matrix from struct for normalization
for i=1:size(PooledFeatures.Features,2)
    NNFeature(i,:)=real(PooledFeatures.Features{i});
end

%if it is training data, compute statistics,
%if not, use the stats provided as input for test data
if strcmp(PooledFeatures.TrainOrTest,'Train')
    NNFeature_mean = mean(NNFeature);
    NNFeature_sd = sqrt(var(NNFeature)+0.01);
else
    NNFeature_mean=DataStats.Mean;
    NNFeature_sd=DataStats.Std;
end
%subtract mean and divide by std
NNFeature = bsxfun(@rdivide, bsxfun(@minus, NNFeature, NNFeature_mean), NNFeature_sd);

if options.Binarize
    NNFeature(NNFeature>options.BinarizationThreshold)=1;
    NNFeature(NNFeature<options.BinarizationThreshold)=0;
end

% %add 1 to the end for minFunc
% NNFeature = [NNFeature, ones(size(NNFeature,1),1)];

%form the struct
NormalizedFeatures.options=options;
NormalizedFeatures.Features=NNFeature;
optionsAll{1}=PooledFeatures.options;
optionsAll{2}=PooledFeatures.optionsPrev;
NormalizedFeatures.optionsPrev=optionsAll;
NormalizedFeatures.optionsAll=optionsAll;

%also include the RF data, for future reference
NormalizedFeatures.RFData=ReceptiveFields;
%bookkeeping of the choices
NormalizedFeatures.SparsifyFlag=PooledFeatures.SparsifyFlag;
NormalizedFeatures.SparsityMultiplier=PooledFeatures.SparsityMultiplier;
NormalizedFeatures.RatioNonZero=PooledFeatures.RatioNonZero;
NormalizedFeatures.TrainOrTest=PooledFeatures.TrainOrTest;
NormalizedFeatures.SummationType=PooledFeatures.SummationType;
NormalizedFeatures.Binarize=options.Binarize;
NormalizedFeatures.Mean=NNFeature_mean;
NormalizedFeatures.Std=NNFeature_sd;

%save if instructed
if options.SaveFeatures
    if strcmp(PooledFeatures.TrainOrTest,'Train')
        FilePrefix='NormalizedFeaturesTrain_';
    else
        FilePrefix='NormalizedFeaturesTest_';
    end
    save(fullfile(options.SavePath,strcat(FilePrefix,ReceptiveFields.DatasetName,'_NoRFs',int2str(numCentroids),'_RFSize',int2str(rfSize),'_Sp',int2str(int8(PooledFeatures.SparsifyFlag)),...
        '_SMp',int2str(100*PooledFeatures.SparsityMultiplier),'_',PooledFeatures.SummationType,'_Bin',int2str(int8(options.Binarize)))),'NormalizedFeatures')
    fprintf('Normalized/Binarized Features saved. \n');
end

%show the finalization on command window
fprintf('Normalize/Binarize done!:   %d \n',toc);

end%end function


