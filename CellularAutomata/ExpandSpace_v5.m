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


function [ExpandedFeatures, options]=ExpandSpace_v5(RawFeatures,DataStats,options)

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
    ExpandedFeatures.W=[];
    ExpandedFeatures.Win=[];
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
if optionsData.Verbose 
    fprintf('Feature Expansion Start: \n'); 
end

%infer knowledge on the RFs
centroids=RawFeatures.RFData.RFs;
numCentroids = size(RawFeatures.RFData.RFs,1);
NumChannels=size(RawFeatures.RFData.RFs,2)/((RawFeatures.RFData.RFSize)^2);
rfSize=RawFeatures.RFData.RFSize;
TotalIterSize=size(RawFeatures.Features,2)*optionsExpansion.CAPerm;
LastIterMask=TotalIterSize*(optionsExpansion.CAIter-1)+1:TotalIterSize*(optionsExpansion.CAIter);
RandomPermutation=[];
SamplingMask=[];
W=optionsExpansion.W;
Win=optionsExpansion.Win;

%prepare some common (and small sized) data structures
if strcmp(optionsExpansion.Type,'CA')
    
    %if the task is a sequence task, we can not use batches in computation, due
    %to the fact that we need to combine subsequent inputs one after another.
    %if ResetSequence is selected, batch computation can be done, inside another loop (see below). 
    if optionsData.DatasetSequenceFlag 
        %input are fetched one-by-one, in the outer loop, because it is a
        %sequence
        optionsExpansion.BatchSize=1;
        
        %but a batch can be formed, and the size is determined by the total
        %size of the dataset
        BatchSizeSequence=floor(size(RawFeatures.Features,1)/optionsExpansion.Treset)-2;
        
        %Whole state matrix, initilize according to computed size
        CAStates_Init = zeros(BatchSizeSequence+1, size(RawFeatures.Features,2)*optionsExpansion.CAPerm*optionsExpansion.CAIter);
        
        %end of the index for computation
        %BEWARE: 30% of the computation is redundant due to warm reservoir.
        %It means that the first 30% of the computation is thrown out.
        EndIndex=round((1+optionsExpansion.WarmUpSequenceRatio)*(optionsExpansion.Treset));
    else
        %Whole state matrix, initilize 
        CAStates_Init = zeros(optionsExpansion.BatchSize, size(RawFeatures.Features,2)*optionsExpansion.CAPerm*optionsExpansion.CAIter);
        
        %end of the index for computation
        EndIndex=size(RawFeatures.Features,1);
        BatchSizeSequence=0;%for recording purposes
    end
    
    %if binarization procedure, created integer outputs, convert them back
    %to binary vectors
    if not(optionsData.BinaryDataset) && strcmp(optionsNormalize.BinarizeType,'QuantizeInteger')
        Bitlength=log2(length(optionsNormalize.QuantizationThresholds)+1);
        
        FeatTemp=RawFeatures.Features';
        FeatTemp=FeatTemp(:);
        FeatTemp=sparse(FeatTemp,1:length(FeatTemp),1);
        FeatTemp=full(FeatTemp)';
        FeatTemp=FeatTemp';
        FeatTemp=reshape(FeatTemp,[size(RawFeatures.Features,2)*(length(optionsNormalize.QuantizationThresholds)+1),size(RawFeatures.Features,1)]);
        FeatTemp=FeatTemp';
        RawFeatures.Features=FeatTemp;
    end
    
    %rule array for CA evolution
    RuleArray = (bitget(optionsExpansion.CARule, 1:8) + 1);
    %state matrix at CA iteration, initialize.
    CAStates_Iter=CAStates_Init(:,1:optionsExpansion.CAPerm*size(RawFeatures.Features,2));
    CAStates_Iter_temp=CAStates_Iter;
    CAImageAll=[];
    
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
    
    if strcmp(optionsExpansion.MappingOperation,'LinearProjection') && isempty(optionsExpansion.W)
        W = optionsExpansion.SR * generate_internal_weights(optionsExpansion.CAPerm*size(RawFeatures.Features,2), optionsExpansion.connectivity);
    end
    
    if strcmp(optionsExpansion.MappingOperation,'LinearProjection') && isempty(optionsExpansion.Win)
        Win = 2 * (rand(optionsExpansion.CAPerm*size(RawFeatures.Features,2), size(RawFeatures.Features,2)) - 0.5) * diag(optionsExpansion.inScaling*ones(size(RawFeatures.Features,2),1));
    end 
   
    
    %make it a binary matrix to avoid multiplication. This is important for
    %computational perspective. (don't binarize if continuous CA)
    if strcmp(optionsExpansion.MappingOperation,'LinearProjection') && isempty(optionsExpansion.Win) && not(optionsExpansion.CAContinuous)
        W=sign(full(W));
        Win=sign(Win);
    end
    
    if isempty(optionsExpansion.Win)
        optionsExpansion.W=W;
        optionsExpansion.Win=Win;
    end

    %put into GPU if chosen
    if optionsExpansion.GPU
        W=gpuArray(full(W));
        Win=gpuArray(Win);
        CAStates_Init=gpuArray(CAStates_Init);
        CAStates_Iter=gpuArray(CAStates_Iter);
        CAStates_Iter_temp=gpuArray(CAStates_Iter_temp);
        RuleArray=gpuArray(RuleArray);
        RandomPermutation=gpuArray(RandomPermutation);
        SamplingMask=gpuArray(SamplingMask);
    end
    
elseif strcmp(optionsExpansion.Type,'ESN')
    %if the task is a sequence task, we can not use batches in computation, due
    %to the fact that we need to combine subsequent inputs one after another.
    %if ResetSequence is selected, batch computation can be done, inside another loop (see below). 
    if optionsData.DatasetSequenceFlag 
        %input are fetched one-by-one, in the outer loop, because it is a
        %sequence
        optionsExpansion.BatchSize=1;
        
        %but a batch can be formed, and the size is determined by the total
        %size of the dataset
        BatchSizeSequence=floor(size(RawFeatures.Features,1)/optionsExpansion.Treset)-2;
           
        %end of the index for computation
        %BEWARE: 30% of the computation is redundant due to warm reservoir.
        %It means that the first 30% of the computation is thrown out.
        EndIndex=round((1+optionsExpansion.WarmUpSequenceRatio)*(optionsExpansion.Treset));
    else
        
        %end of the index for computation
        EndIndex=size(RawFeatures.Features,1);
        BatchSizeSequence=0;%for recording purposes
    end
    
    %initialize state of the network.
    CAStates = zeros(optionsExpansion.nInternalUnits,BatchSizeSequence+1);
    
    if isempty(optionsExpansion.W)
        W = optionsExpansion.SR * generate_internal_weights(optionsExpansion.nInternalUnits, optionsExpansion.connectivity);
    end
    
    if isempty(optionsExpansion.Win)
        Win = 2 * (rand(optionsExpansion.nInternalUnits, size(RawFeatures.Features,2)) - 0.5) * diag(optionsExpansion.inScaling*ones(size(RawFeatures.Features,2),1));
    end
    
% %     %DEBUG CODE:REMOVE LATER!!!
%     W=sign(full(W));%sign(full(W)-0.001);
%     Win=sign(Win);
    
    optionsExpansion.W=W;
    optionsExpansion.Win=Win;

    %put into GPU if chosen
    if optionsExpansion.GPU
        CAStates=gpuArray(CAStates);
        W=gpuArray(full(W));
        Win=gpuArray(Win);
    end
    
end %end of expansion type conditional


%process each image in batches of size:BatchSize,  in a for loop.
%Batch processing exploits GPU power. Because single image is just not
%large enough for most of the experiments (eg. CIFAR10). 
for i=1:optionsExpansion.BatchSize:EndIndex
    
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
        
        %Some explanation of the variables:
        %RuleArray: the rule used in CA, 
        %RandomPermutation: the random permutations used for state space
        %   generation
        %SamplingMask: some of the states are kept for brevity.
       
        
        %if the task is not a sequence task, process each datapoint
        %individually (batch computation is not possible).
        % Else, combine subsequent datapoints using an operation (i.e. XOR, Normalized summation).
        if not(optionsData.DatasetSequenceFlag)
            SamplePattern=i:i+optionsExpansion.BatchSize-1;
            InitialStates=RawFeatures.Features(SamplePattern,:);
            
            if optionsExpansion.GPU
                InitialStates=gpuArray(InitialStates);
            end
            
            if strcmp(optionsExpansion.MappingOperation,'LinearProjection')
                CAStates_Iter=Win * InitialStates';
                CAStates_Iter(CAStates_Iter>optionsExpansion.LinearProjectionThreshold)=1;
                CAStates_Iter(CAStates_Iter<optionsExpansion.LinearProjectionThreshold)=0;
                CAStates_Iter=CAStates_Iter';
            else 
                %put every random permutation one after another, create a big redundant
                %feature vector (named CAStates_Iter)
                for pe=1:1:optionsExpansion.CAPerm
                    RandPermutTemp=RandomPermutation(pe,1:size(RawFeatures.Features,2));
                    InitialStates_=InitialStates(:,RandPermutTemp);
                    CAStates_Iter(:,(pe-1)*size(RawFeatures.Features,2)+1:pe*size(RawFeatures.Features,2))=InitialStates_; %add 1 for Ele. CA processing
                end
            end
        else
            %if the CA states are reset every Treset, then process in
            %batches
            if optionsExpansion.ResetSequence
                SamplePattern=i:optionsExpansion.Treset:i+optionsExpansion.Treset*BatchSizeSequence;
                CurrentInput=RawFeatures.Features(SamplePattern,:); %current input in the sequence
            else
                SamplePattern=i; %if no reser, then always fetch input one-by-one
                CurrentInput=RawFeatures.Features(SamplePattern,:);
            end
            
            if optionsExpansion.GPU
                CurrentInput=gpuArray(CurrentInput);
            end
            
            if strcmp(optionsExpansion.MappingOperation,'LinearProjection')
                CAStates_Iter=Win * CurrentInput';
                if not(optionsExpansion.CAContinuous)
                    CAStates_Iter(CAStates_Iter-optionsExpansion.LinearProjectionThreshold>0)=1;
                    CAStates_Iter(CAStates_Iter-optionsExpansion.LinearProjectionThreshold<=0)=0;
                end
                CAStates_Iter=CAStates_Iter';
            else 
                %put every random permutation one after another, create a big redundant
                %feature vector (named CAStates_Iter)
                for pe=1:1:optionsExpansion.CAPerm
                    RandPermutTemp=RandomPermutation(pe,1:size(RawFeatures.Features,2));
                    CurrentInput_=CurrentInput(:,RandPermutTemp);
                    CAStates_Iter(:,(pe-1)*size(RawFeatures.Features,2)+1:pe*size(RawFeatures.Features,2))=CurrentInput_; 
                end
            end
            
            %combine the last reservoir state with the current input via
            %XOR or normalized summation operations.
            if i>1 %combination occurs after the t=1 in sequence
                if strcmp(optionsExpansion.CombinationOperation,'XOR')
                    CAStates_Iter=xor(logical(CAStates(:,LastIterMask)),logical(CAStates_Iter));
                elseif  strcmp(optionsExpansion.CombinationOperation,'NormalizedSummation') %normalized summation
                    
%                     %feedback to input .
%                     CurrentInput=CurrentInput+CAStates(:,LastIterMask)*Win;
%                     CAStates_Iter=Win * CurrentInput';
%                     CAStates_Iter = bsxfun(@rdivide, bsxfun(@minus, CAStates_Iter, mean(CAStates_Iter,2)), sqrt(var(CAStates_Iter,[],2))+realmin);
%                     CAStates_Iter(CAStates_Iter>optionsExpansion.LinearProjectionThreshold)=1;
%                     CAStates_Iter(CAStates_Iter<optionsExpansion.LinearProjectionThreshold)=0;
%                     CAStates_Iter=CAStates_Iter';
                    
%                     %two item normalized summation, weighted
%                     Sum=0.5*CAStates_Iter + 0.5*CAStates(:,LastIterMask);
%                     CAStates_Iter(Sum>0.5)=1;
%                     CAStates_Iter(Sum<0.5)=(rand(gather(sum(Sum(:)<0.5)),1)>0.5);
                    
%                     %three item normalized summation. Very similar to ESN
%                     %network equations.
%                     Sum=CAStates(:,LastIterMask)+ CAStates_Iter; %S_3temp = S_2^I + S_3
%                     CAStates_Iter_temp(Sum==2)=1;
%                     CAStates_Iter_temp(Sum==0)=0;
%                     CAStates_Iter_temp(Sum==1)=(rand(gather(sum(Sum(:)==1)),1)>0.5);            
%                     Sum2=CAStates_PrevReservoir + CAStates_Iter_temp; %S_3^+ =  S_2^+ S_3temp 
%                     CAStates_Iter(Sum2==2)=1;
%                     CAStates_Iter(Sum2==0)=0;
%                     CAStates_Iter(Sum2==1)=(rand(gather(sum(Sum2(:)==1)),1)>0.5);

                    if optionsExpansion.CAContinuous
                        %fractional remainder summation for continuous CA
%                         Sum= CAStates_Iter + CAStates(:,LastIterMask); %+ optionsExpansion.ContinuousCASumTerm;
                        Sum=optionsExpansion.ContinuousCAWeightNewSeq*CAStates_Iter ...
                            + CAStates(:,LastIterMask); %+ optionsExpansion.ContinuousCASumTerm;
                        CAStates_Iter = Sum - floor(Sum);
%                         CAStates_Iter = Sum;
                    else
                        %two item normalized summation
                        Sum=CAStates_Iter + CAStates(:,LastIterMask);
                        CAStates_Iter(Sum==2)=1;
                        CAStates_Iter(Sum==0)=0;
                        CAStates_Iter(Sum==1)=(rand(gather(sum(Sum(:)==1)),1)>0.5);
                    end

                elseif  strcmp(optionsExpansion.CombinationOperation,'ESNType') %normalized summation
                    %ESN type real computation
                    CAStates_Iter=Win * CurrentInput' + W*CAStates(:,LastIterMask)';
                    CAStates_Iter(CAStates_Iter-optionsExpansion.LinearProjectionThreshold>0)=1;
                    CAStates_Iter(CAStates_Iter-optionsExpansion.LinearProjectionThreshold<=0)=0;
                    CAStates_Iter=CAStates_Iter';

                    
                end
            end
           %record S_i^+ (previous reservoir state).
           CAStates_PrevReservoir=CAStates_Iter;

        end
        
        %evaluate CA cell states for a number of iterations.
        CAStates = CellularAutomataEvolution_v3(CAStates_Iter, CAStates_Init, RuleArray, RandomPermutation, SamplingMask, optionsExpansion);
        
        %save the states of one example for display
        CAImage=CAStates(1,:);
        CAImage=(reshape(CAImage',[size(CAStates(1,:),2)/optionsExpansion.CAIter optionsExpansion.CAIter ]))';
        CAImageAll=[CAImageAll; CAImage ];
        
    elseif strcmp(optionsExpansion.Type,'ESN')
        %if the task is not a sequence task, process each datapoint
        %individually (batch computation is not possible).
        % Else, combine subsequent datapoints using an operation (i.e. XOR, Normalized summation).
        if not(optionsData.DatasetSequenceFlag)
            SamplePattern=i:i+optionsExpansion.BatchSize-1;
            CurrentInput=RawFeatures.Features(SamplePattern,:);
            
            if optionsExpansion.GPU
                CurrentInput=gpuArray(CurrentInput);
            end
            
        else
            %if the CA states are reset every Treset, then process in
            %batches
            if optionsExpansion.ResetSequence
                SamplePattern=i:optionsExpansion.Treset:i+optionsExpansion.Treset*BatchSizeSequence;
                CurrentInput=RawFeatures.Features(SamplePattern,:); %current input in the sequence
            else
                SamplePattern=i; %if no reser, then always fetch input one-by-one
                CurrentInput=RawFeatures.Features(SamplePattern,:);
            end
            
            if optionsExpansion.GPU
                CurrentInput=gpuArray(CurrentInput);
            end
            
            
        end
        
        %format changes between CA and ESN causes these transposes.
        if i>1
            CAStates=CAStates';
        end
        
        CAStates = (1-optionsExpansion.LR) * CAStates + optionsExpansion.LR * tanh(W * CAStates + Win * CurrentInput');  
        
%         %DEBUG CODE:REMOVE LATER!!!
%         if i==1
%             CAStates = (1-optionsExpansion.LR) * CAStates + optionsExpansion.LR * sign((W * CAStates + Win * CurrentInput')); 
%         else
%             CAStates = (1-optionsExpansion.LR) * CAStates + optionsExpansion.LR * sign((W * CAStates + Win * CurrentInput')); 
%         end
        
        CAStates=CAStates';
         
        %TODO: Autopilot mode. Run without input for expansion.
        
    else
        disp('Not proper expansion option!')
        ExpandedFeatures=RawFeatures;
        ExpandedFeatures.optionsAll=options;
        ExpandedFeatures.optionsPrev=options;
        return
    end
    
    
    ExpandedFeatures.Features(SamplePattern,:) = gather(CAStates);
end

%pass the label info on the expanded feature space
ExpandedFeatures.Labels=RawFeatures.Labels;
%some of the data points are chopped off due to sequence processing,
if optionsData.DatasetSequenceFlag
    ExpandedFeatures.Labels=ExpandedFeatures.Labels(1:size(ExpandedFeatures.Features),:);
end
ExpandedFeatures.UniqueLabels=RawFeatures.UniqueLabels;

%show states
% imagesc(CAImageAll)
% colormap(flipud(gray))
% imwrite(max(CAImageAll(:))-CAImageAll,'myGray.png','png');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%NORMALIZATION START 

%convert to integer for better classifier training. Binary data
%is not a good choice for linear classifiers.
if optionsExpansion.ConvertInteger
    BitSize=optionsExpansion.IntegerBitSize;
    FeatTemp=ExpandedFeatures.Features';
    FeatTemp=FeatTemp(:);
    FeatTemp=reshape(FeatTemp,[BitSize size(ExpandedFeatures.Features,1)*size(ExpandedFeatures.Features,2)/BitSize]);
    FeatTemp=FeatTemp';
    FeatTemp=bi2de(FeatTemp);
    FeatTemp=reshape(FeatTemp,[size(ExpandedFeatures.Features,2)/BitSize size(ExpandedFeatures.Features,1)]);
    FeatTemp=FeatTemp';
    ExpandedFeatures.Features=FeatTemp;
end

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
options{5}=optionsExpansion;
ExpandedFeatures.optionsAll=options;
ExpandedFeatures.optionsPrev=options;
ExpandedFeatures.BatchSizeSequence=ExpandedFeatures;
ExpandedFeatures.RFData=RawFeatures.RFData;
ExpandedFeatures.Mean=RawFeatures.Mean;
ExpandedFeatures.Std=RawFeatures.Std;
ExpandedFeatures.ExMean=ExpandedFeatures_mean;
ExpandedFeatures.ExStd=ExpandedFeatures_sd;
ExpandedFeatures.W=W;
ExpandedFeatures.Win=Win;
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
        '_SMp',int2str(100*optionsExtract.SparsityMultiplier),'_',optionsPool.SummationType,'_Bin',int2str(int8(optionsNormalize.Binarize)),'E',int2str(int8(optionsExpansion.Perform)),'ET',optionsExpansion.Type)),'ExpandedFeatures','-v7.3')
    if optionsData.Verbose 
        fprintf('Expanded Features saved. \n'); 
    end
end


%show the finalization on command window
if optionsData.Verbose 
    fprintf('Feature Expansion done! % d:   \n', toc); 
end

end%end function


