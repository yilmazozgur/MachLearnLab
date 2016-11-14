%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function expands the space by  reservoir computing or cellular
%automata
%
%%% Input:
%RawFeatures : A struct of the features directly from data or after some
%   feature computation.
%DataStats:statistics of data for normalization
%options: struct for options. It inherits all the options from previous
%   stages, i.e. extraction, pooling and normalization/binarization.
%   Task specific options are:
% optionsExpansion.Perform: Flag that determines whether there is expansion.
%   Default=false;
% optionsExpansion.Type: The type of expansion. Default='CA';  
% optionsExpansion.CARule: Rule for cellular automata. Default=90;
% optionsExpansion.CAIter: Number of CA evolutions. Default=4;
% optionsExpansion.CAPerm: Numbe of random permutations. Default=4;
% optionsExpansion.SamplingRatio: The expanded space is subsampled for brevity. 
%   Default=8*1/(optionsExpansion.CAIter*optionsExpansion.CAPerm);
% optionsExpansion.Normalize: Flag for mean and std normalization. Default=false;
% optionsExpansion.GPU: GPU computing flag. Default=true;
% optionsExpansion.BatchSize: GPU is exploited better if processed in batches.
%   This is batch size. Default =100;
% optionsExpansion.SaveFeatures: Default=false;
% optionsExpansion.SavePath;
%
%%% Output:
%ExpandedFeatures: A struct that holds the expanded features for each image.
%
%
%From:
%TOU_ML
%Ozgur Yilmaz, Turgut Ozal University, Ankara
%Web: ozguryilmazresearch.net
%Ref: It is based on Adam Coates code, Coates and Ng, 2011 paper.
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function ExpandedFeatures=ExpandSpace(RawFeatures,DataStats,options)

%load options of every stage in a separate struct
optionsData=options{1};
optionsExtract=options{2};
optionsPool=options{3};
optionsNormalize=options{4};
optionsExpansion=options{5};

%if the precomputed expanded features, or model are loaded during LoadDataset,
%return from the function.
if strcmp(optionsData.WhichData,'ExpandedFeatures') || strcmp(optionsData.WhichData,'Model')
    ExpandedFeatures=RawFeatures;
    if strcmp(optionsData.WhichData,'ExpandedFeatures')
        ExpandedFeatures.ExMean=RawFeatures.ExMean;
        ExpandedFeatures.ExStd=RawFeatures.ExStd;
        ExpandedFeatures.RandomPermutation=RawFeatures.RandomPermutation;
        ExpandedFeatures.SamplingMask=RawFeatures.RandomPermutation;
    end
    return
end

%if no expansion is needed, return. 
%This is necessary not to make a major change in the experiment script.
if not(optionsExpansion.Perform)
    ExpandedFeatures=RawFeatures;
    ExpandedFeatures.Labels=RawFeatures.Labels;
    ExpandedFeatures.UniqueLabels=RawFeatures.UniqueLabels;
    ExpandedFeatures.optionsAll=options;
    ExpandedFeatures.optionsPrev=options;
    ExpandedFeatures.ExMean=[];
    ExpandedFeatures.ExStd=[];
    ExpandedFeatures.RandomPermutation=[];
    ExpandedFeatures.SamplingMask=[];

    return
end

%default options, if not given by the user
if nargin < 3
    optionsExtract.SparsifyFlag=true;
    optionsExtract.SparsityMultiplier=1;
    optionsExtract.GPU=true;
    optionsExtract.BatchSize=20;
    optionsExtract.TrainOrTest='Train';
    optionsExtract.SaveFeatures=false;
    optionsExtract.SavePath='F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\SavedData';
    
    optionsPool.SummationType='Coarse';
    optionsPool.GPU=false;
    optionsPool.SaveFeatures=false;
    optionsPool.SavePath='F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\SavedData';

    optionsNormalize.Binarize=true;
    optionsNormalize.BinarizationThreshold=0;
    optionsNormalize.GPU=false;
    optionsNormalize.SaveFeatures=false;
    optionsNormalize.SavePath='F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\SavedData';

    optionsExpansion.Perform=false;
    optionsExpansion.Type='CA'; %'CA'
    optionsExpansion.CARule=90;
    optionsExpansion.CAIter=4;
    optionsExpansion.CAPerm=4;
    optionsExpansion.SamplingRatio=4*1/(optionsExpansion.CAIter*optionsExpansion.CAPerm);
    optionsExpansion.Normalize=false;
    optionsExpansion.GPU=true;
    optionsExpansion.BatchSize=100;
    optionsExpansion.SaveFeatures=false;
    optionsExpansion.SavePath='F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\SavedData';

end


%measure computation time
tic

%print what is to be done
fprintf('Feature Expansion Start: \n');

%infer knowledge on the RFs
centroids=RawFeatures.RFData.RFs;
numCentroids = size(RawFeatures.RFData.RFs,1);
NumChannels=size(RawFeatures.RFData.RFs,2)/((RawFeatures.RFData.RFSize)^2);
rfSize=RawFeatures.RFData.RFSize;


%prepare some common (and small sized) data structures
if strcmp(optionsExpansion.Type,'CA')
    %rule array for CA evolution
    RuleArray = (bitget(optionsExpansion.CARule, 1:8) + 1);
    %Whole state matrix, initilize once
    CAStates_Init = zeros(optionsExpansion.BatchSize, size(RawFeatures.Features,2)*optionsExpansion.CAPerm*optionsExpansion.CAIter);

    %if it is traininf data, create a random permutation matrix and sampling mask,
    %if not, use existing
    if strcmp(optionsData.TrainTest,'Train') && isempty(DataStats)
        RandomPermutation=zeros(optionsExpansion.CAPerm,size(RawFeatures.Features,2));
        for j=1:1:optionsExpansion.CAPerm
%             RandomPermutation(j,:)=1:size(RawFeatures.Features,2);
            RandomPermutation(j,:)=randperm(size(RawFeatures.Features,2));
        end
        if optionsExpansion.SamplingRatio>0
            %Whole feature space size
            WholeSpaceSize=size(RawFeatures.Features,2)*optionsExpansion.CAIter*optionsExpansion.CAPerm;
            %how many samples are used
            NoOfSamples=round(WholeSpaceSize*optionsExpansion.SamplingRatio);
            RandVector=randperm(WholeSpaceSize);
            SamplingMask=RandVector(1:NoOfSamples);
        else
            SamplingMask=[];
        end
    else
        RandomPermutation=DataStats.RandomPermutation;
        SamplingMask=DataStats.SamplingMask;
    end
    
    %put into GPU if chosen
    if optionsExpansion.GPU
        CAStates_Init=gpuArray(CAStates_Init);
        RuleArray=gpuArray(RuleArray);
        RandomPermutation=gpuArray(RandomPermutation);
        SamplingMask=gpuArray(SamplingMask);
    end
    
end

%process each image in batches of size:BatchSize,  in a for loop.
%Batch processing exploits GPU power. Because single image is just not
%large enough for most of the experiments (eg. CIFAR10). 
for i=1:optionsExpansion.BatchSize:size(RawFeatures.Features,1)
    
    %if CA expansion is chosen, call the appropriate function
    if strcmp(optionsExpansion.Type,'CA')
        
        %if the data is not binary, then display an error message and exit.
        if not(optionsNormalize.Binarize) && not(options{1}.BinaryDataset)
            disp('Cannot do CA expansion on non-binary data.')
            ExpandedFeatures=RawFeatures;
            ExpandedFeatures.optionsAll=options;
            ExpandedFeatures.optionsPrev=options;
            ExpandedFeatures.ExMean=[];
            ExpandedFeatures.ExStd=[];
            ExpandedFeatures.RandomPermutation=[];
            ExpandedFeatures.SamplingMask=[];
            return
        end
        
        %RuleArray: the rule used in CA, 
        %RandomPermutation: the random permutations used for state space
        %   generation
        %SamplingMask: some of the states are kept for brevity.
        InitialStates=RawFeatures.Features(i:i+optionsExpansion.BatchSize-1,:);
        if optionsExpansion.GPU
            InitialStates=gpuArray(InitialStates);
        end
%         optionsExpansion.CAIter=0;
        CAStates = CellularAutomataEvolution(InitialStates, CAStates_Init, RuleArray, RandomPermutation, SamplingMask, optionsExpansion);

    else
        disp('Not proper expansion option!')
        ExpandedFeatures=RawFeatures;
        ExpandedFeatures.optionsAll=options;
        ExpandedFeatures.optionsPrev=options;
        return
    end

    ExpandedFeatures.Features(i:i+optionsExpansion.BatchSize-1,:) = gather(CAStates);
end

%pass the label info on the expanded feature space
ExpandedFeatures.Labels=RawFeatures.Labels;
ExpandedFeatures.UniqueLabels=RawFeatures.UniqueLabels;

%if Normalization flag is ON, do mean and std normalization.
if optionsExpansion.Normalize
    %if it is training data, compute statistics,
    %if not, use the stats provided as input for test data
    if strcmp(optionsData.TrainTest,'Train')
        if isempty(DataStats)
            ExpandedFeatures_mean = mean(ExpandedFeatures.Features);
            ExpandedFeatures_sd = sqrt(var(ExpandedFeatures.Features)+0.01);
        else
            ExpandedFeatures_mean = (mean(ExpandedFeatures.Features)+DataStats.ExMean)/2;
            ExpandedFeatures_sd = (sqrt(var(ExpandedFeatures.Features)+0.01)+DataStats.ExStd)/2;
        end
    else
        ExpandedFeatures_mean=DataStats.ExMean;
        ExpandedFeatures_sd=DataStats.ExStd;
    end
    %subtract mean and divide by std
    ExpandedFeatures.Features = bsxfun(@rdivide, bsxfun(@minus, ExpandedFeatures.Features, ExpandedFeatures_mean), ExpandedFeatures_sd);
else
    ExpandedFeatures_mean=[];
    ExpandedFeatures_sd=[];
end

%NORMALIZATION END    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%also include the RF data and all options, for future reference
ExpandedFeatures.optionsAll=options;
ExpandedFeatures.optionsPrev=options;
ExpandedFeatures.RFData=RawFeatures.RFData;
ExpandedFeatures.Mean=RawFeatures.Mean;
ExpandedFeatures.Std=RawFeatures.Std;
ExpandedFeatures.ExMean=ExpandedFeatures_mean;
ExpandedFeatures.ExStd=ExpandedFeatures_sd;
ExpandedFeatures.RandomPermutation=gather(RandomPermutation);
ExpandedFeatures.SamplingMask=gather(SamplingMask);
ExpandedFeatures.SparsifyFlag=optionsExtract.SparsifyFlag;
ExpandedFeatures.SparsityMultiplier=optionsExtract.SparsityMultiplier;
ExpandedFeatures.SummationType=optionsPool.SummationType;
ExpandedFeatures.Binarize=optionsNormalize.Binarize;
ExpandedFeatures.DatasetName=RawFeatures.DatasetName;

%save if instructed
if optionsExpansion.SaveFeatures && optionsExpansion.Perform
    if strcmp(optionsData.TrainTest,'Train')
        FilePrefix=strcat('ExpandedFeaturesTrain_',int2str(optionsData.BatchNumber),'_');
    else
        FilePrefix=strcat('ExpandedFeaturesTest_',int2str(optionsData.BatchNumber),'_');
    end
    save(fullfile(optionsExpansion.SavePath,strcat(FilePrefix,RawFeatures.DatasetName,'_NoRFs',int2str(numCentroids),'_RFSize',int2str(rfSize),'_Sp',int2str(int8(optionsExtract.SparsifyFlag)),...
        '_SMp',int2str(100*optionsExtract.SparsityMultiplier),'_',optionsPool.SummationType,'_Bin',int2str(int8(optionsNormalize.Binarize)),'E',int2str(int8(optionsExpansion.Perform)),'ET',optionsExpansion.Type)),'ExpandedFeatures')
    fprintf('Expanded Features saved. \n');
end


%show the finalization on command window
fprintf('Feature Expansion done! % d:   \n', toc);

end%end function


