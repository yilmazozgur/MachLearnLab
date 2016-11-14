%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function computed single layer NN activities to be used 
%for vision/machine learning tasks.Version 2 (v2) does batch image
%processing, hence is able to use GPU more efficiently.
%
%%% Input:
%Dataset : A struct of the dataset created by LoadDataset.m
%ReceptiveFields: A struct, that has centroids of RFs, mean and covariance 
%   matrices.
%DataStats:Statistics of data for normalization. Mean and Std normalization
%   is performed using this. 
%options: struct for options. It includes the options for three processses:
%   extract, pool and normalize.
%
%optionsExtract.SparsifyFlag: flag for sparsification of neural responses. 
%   Default=true.
%optionsExtract.SparsityMultiplier: the amounf of sparsity. Default=1 (50% nonzero).
%optionsExtract.GPU: enable GPU computing. Default=false.
%optionsExtract.BatchSize: the number of images to be processed together. Default=20;
%optionsExtract.TrainOrTest: Train or Test data will be worked on. Default: 'Train'.
%optionsExtract.SaveFeatures: Save the results to the Root when finished.
%   Default=false.
%optionsExtract.SavePath: the path to save the results if SaveRFs flag is up.
%   Default='';
%
%optionsPool.SummationType: The number of regions to be pooled is determined 
%   by this option. 4by4 'Fine', 2by2 'Coarse' or 1by1 'Single' reductions.
%   Default='coarse';
%optionsPool.GPU: enable GPU computing. Default=false.
%optionsPool.SaveFeatures: Save the results to the Root when finished.
%   Default=false.
%optionsPool.SavePath: the path to save the results if SaveRFs flag is up.
%   Default='';
%
%optionsNormalize.Binarize: a flag that determines if the output feature vector will
%   be binary.
%optionsNormalize.BinarizationThreshold: After mean and std normalization, 
%   this threshold determines which entries are non-zero in binary vector. Default=0;
%optionsNormalize.GPU: enable GPU computing. Default=false.
%optionsNormalize.SaveFeatures: Save the results to the Root when finished.
%   Default=false.
%optionsNormalize.SavePath: the path to save the results if SaveRFs flag is up.
%   Default='';

%%% Output:
%NeuralNetFeatures: A struct that holds the pooled and normalized single
%layer neural network feature of the data.
%
%
%From:
%TOU_ML
%Ozgur Yilmaz, Turgut Ozal University, Ankara
%Web: ozguryilmazresearch.net
%Ref: It is based on Adam Coates code, Coates and Ng, 2011 paper.
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function NeuralNetFeatures=ExtractPoolNormalizeSingleLayerFeature_v2(Dataset,ReceptiveFields,DataStats,options)

%load options of every stage in a separate struct
optionsData=options{1};
optionsExtract=options{2};
optionsPool=options{3};
optionsNormalize=options{4};

%if the precomputed features, or model are loaded during LoadDataset,
%return from the function.
if strcmp(optionsData.WhichData,'Features') ||  strcmp(optionsData.WhichData,'ExpandedFeatures') || strcmp(optionsData.WhichData,'Model')
    NeuralNetFeatures=Dataset;
    return
end


%if no expansion is needed, return. 
%This is necessary not to make a major change in the experiment script.
if not(optionsExtract.Perform)
    if strcmp(Dataset.options{1,1}.TrainTest,'Train')
        NeuralNetFeatures.Features=Dataset.trainX;
        NeuralNetFeatures.Labels=Dataset.trainY;
    else
        NeuralNetFeatures.Features=Dataset.testX;
        NeuralNetFeatures.Labels=Dataset.testY;
    end
    NeuralNetFeatures.UniqueLabels=Dataset.UniqueLabels;
    NeuralNetFeatures.optionsAll=options;
    NeuralNetFeatures.optionsPrev=options;    
    NeuralNetFeatures.RFData=ReceptiveFields;
    NeuralNetFeatures.SparsifyFlag=optionsExtract.SparsifyFlag;
    NeuralNetFeatures.SparsityMultiplier=optionsExtract.SparsityMultiplier;
    NeuralNetFeatures.SummationType=optionsPool.SummationType;
    NeuralNetFeatures.Binarize=optionsNormalize.Binarize;
    NeuralNetFeatures.DatasetName=Dataset.DatasetName;

    if strcmp(optionsData.TrainTest,'Train')
        if isempty(DataStats)
            XPooled_mean = mean(NeuralNetFeatures.Features);
            XPooled_sd = sqrt(var(NeuralNetFeatures.Features)+0.01);
        else %for batch based processing, combine with previous batch stats
            XPooled_mean = (mean(NeuralNetFeatures.Features)+DataStats.Mean)/2;
            XPooled_sd = (sqrt(var(NeuralNetFeatures.Features)+0.01)+DataStats.Std)/2;
        end
    else
        XPooled_mean=DataStats.Mean;
        XPooled_sd=DataStats.Std;
    end
    
    NeuralNetFeatures.Mean=XPooled_mean;
    NeuralNetFeatures.Std=XPooled_sd;

    return
end

%default options, if not given by the user
if nargin < 4
    optionsExtract.Perform=true;
    optionsExtract.SparsifyFlag=true;
    optionsExtract.SparsityMultiplier=1;
    optionsExtract.GPU=true;
    optionsExtract.BatchSize=20;
    optionsExtract.TrainOrTest='Train';
    optionsExtract.SaveFeatures=false;
    optionsExtract.SavePath='';
    
    optionsPool.SummationType='Coarse';
    optionsPool.GPU=false;
    optionsPool.SaveFeatures=false;
    optionsPool.SavePath='';

    optionsNormalize.Binarize=true;
    optionsNormalize.BinarizationThreshold=0;
    optionsNormalize.GPU=false;
    optionsNormalize.SaveFeatures=false;
    optionsNormalize.SavePath='';

end
       

%extract the raw data and image dimensions
if strcmp(optionsData.TrainTest,'Train') %either Train or Test data is processed.
    X=Dataset.trainX; %data
    NeuralNetFeatures.Labels=Dataset.trainY;
else
    X=Dataset.testX; %data
    NeuralNetFeatures.Labels=Dataset.testY;
end
NeuralNetFeatures.UniqueLabels=Dataset.UniqueLabels; %save unique labels from the dataset
ImageDim=Dataset.ImageDim;
OneImageSize=ImageDim(1)*ImageDim(2);

%infer knowledge on the RFs
centroids=single(ReceptiveFields.RFs);
cc = sum(centroids.^2, 2)';
numCentroids = size(ReceptiveFields.RFs,1);
NumChannels=size(ReceptiveFields.RFs,2)/((ReceptiveFields.RFSize)^2);
rfSize=ReceptiveFields.RFSize;

%number of rows and columns in the feature space
prows = ImageDim(1)-rfSize+1;
pcols = ImageDim(2)-rfSize+1;

%measure computation time
tic

%initialize important variable
FeatureImage=zeros(optionsExtract.BatchSize,prows,pcols,numCentroids,'single');
patches=zeros(prows*pcols*optionsExtract.BatchSize,size(centroids,2),'single');
MeanRF=single(ReceptiveFields.M);
StdRF=single(ReceptiveFields.P);

%if GPU processing put data on GPU
if optionsExtract.GPU
%     X=gpuArray(single(X));
    centroids=gpuArray(centroids);
    cc=gpuArray(cc);
    FeatureImage=gpuArray(FeatureImage);
    MeanRF=gpuArray(MeanRF);
    StdRF=gpuArray(StdRF);
%     patches=gpuArray(single(patches));
end


%print what is to be done
fprintf('Feature Extraction, Pooling, Normalization Start: \n');

%process each image in batches of size:BatchSize,  in a for loop.
%Batch processing exploits GPU power. Because single image is just not
%large enough for most of the experiments (eg. CIFAR10). 
for i=1:optionsExtract.BatchSize:size(X,1)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FIND HIDDEN LAYER ACTIVITIES, START
    %put batches in a matrix for fast GPU processing.
    ImageNo=1;
    for j=i:1:i+optionsExtract.BatchSize-1
        % extract overlapping sub-patches into rows of 'patches'
        if ReceptiveFields.ImageIsGrayFlag
            patches_ = (im2col(reshape(X(j,1:OneImageSize),ImageDim(1:2)), [rfSize rfSize]))';
        else
            if ReceptiveFields.GrayConvertFlag
                patches_ = (sqrt(im2col(reshape(X(j,1:OneImageSize),ImageDim(1:2)), [rfSize rfSize]).^2+...
                    im2col(reshape(X(j,OneImageSize+1:2*OneImageSize),ImageDim(1:2)), [rfSize rfSize]).^2 +...
                    im2col(reshape(X(j,2*OneImageSize+1:end),ImageDim(1:2)), [rfSize rfSize]).^2) )';
            else      
                patches_ = [ im2col(reshape(X(j,1:OneImageSize),ImageDim(1:2)), [rfSize rfSize]) ;
                    im2col(reshape(X(j,OneImageSize+1:2*OneImageSize),ImageDim(1:2)), [rfSize rfSize]) ;
                    im2col(reshape(X(j,2*OneImageSize+1:end),ImageDim(1:2)), [rfSize rfSize]) ]';
            end
        end
        patches((ImageNo-1)*size(patches_,1)+1:ImageNo*size(patches_,1),:)=patches_;
        ImageNo=ImageNo+1;
    end
    
    %if GPU processing put data on GPU
    if optionsExtract.GPU
        patches=gpuArray(patches);
    end
    
    % normalize for contrast
    patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

    % whiten
    patches = bsxfun(@minus, patches, MeanRF) * StdRF;
    
    % compute 'triangle' activation function
    xx = sum(patches.^2, 2);
%     cc = sum(centroids.^2, 2)';
    xc = patches * centroids';

    z = sqrt( bsxfun(@plus, cc, bsxfun(@minus, xx, 2*xc)) ); % distances

    if optionsExtract.SparsifyFlag
        mu = mean(z, 2); % average distance to centroids for each patch
        patches2 = max(bsxfun(@minus, mu*optionsExtract.SparsityMultiplier, z), 0);
    else
        patches2=z;
    end % patches is now the data matrix of activations for each patch

    % reshape to numCentroids-channel image
    for kk=1:1:optionsExtract.BatchSize
        FeatureImage(kk,:,:,:)=reshape(patches2((kk-1)*size(patches_,1)+1:kk*size(patches_,1),:),prows, pcols, numCentroids);
    end
%     FeatureImage=reshape(patches2, optionsExtract.BatchSize, prows, pcols, numCentroids);
%     %gather data if GPU is on
%     if optionsExtract.GPU
%         FeatureImage=gather(FeatureImage);
%     end
    
%FIND HIDDEN LAYER ACTIVITIES, END    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%POOL ACTIVITIES, START    
    % find rows and cols
    prows = ImageDim(1)-rfSize+1;
    pcols = ImageDim(2)-rfSize+1;

    % compute half sizes
    halfr = round(prows/2);
    halfc = round(pcols/2);

    %halfhalf sizes
    halfrhalf=round(halfr/2);
    halfchalf=round(halfc/2);

    %depending on the choice, get 8by8, 4by4, 2by2 or 1by1 (whole) or 'None' reductions
    if strcmp(optionsPool.SummationType,'Finest')
        XPooled1=HalveHalveAndPool(FeatureImage(:,1:halfr,1:halfc,:));
        XPooled2=HalveHalveAndPool(FeatureImage(:,halfr+1:end,1:halfc,:));
        XPooled3=HalveHalveAndPool(FeatureImage(:,1:halfr,halfc+1:end,:));
        XPooled4=HalveHalveAndPool(FeatureImage(:,halfr+1:end,halfc+1:end,:));
        XPooled=[XPooled1 XPooled2 XPooled3 XPooled4];
    elseif strcmp(optionsPool.SummationType,'Fine')
        XPooled=HalveHalveAndPool(FeatureImage);
    elseif strcmp(optionsPool.SummationType,'Coarse')
        XPooled=HalveAndPool(FeatureImage);
    elseif strcmp(optionsPool.SummationType,'Single')
        q_sum = squeeze(sum(sum(FeatureImage, 2),3));
        XPooled = [squeeze(q_sum)];
    elseif strcmp(optionsPool.SummationType,'None')
        XPooled=reshape(FeatureImage,[size(FeatureImage,1) size(FeatureImage,2)*size(FeatureImage,3)*size(FeatureImage,4)]);
    end
        

%POOL ACTIVITIES, END    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if optionsExtract.GPU
        XPooled=gather(XPooled);
    end

    %save the feature for each image, in a list
    NeuralNetFeatures.Features(i:i+optionsExtract.BatchSize-1,:) = double(XPooled);
    XPooled=gpuArray(XPooled);
    

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%NORMALIZATION/BINARIZATION, START   

%if it is training data, compute statistics,
%if not, use the stats provided as input for test data
if strcmp(optionsData.TrainTest,'Train')
    if isempty(DataStats)
        XPooled_mean = mean(NeuralNetFeatures.Features);
        XPooled_sd = sqrt(var(NeuralNetFeatures.Features)+0.01);
    else %for batch based processing, combine with previous batch stats
        XPooled_mean = (mean(NeuralNetFeatures.Features)+DataStats.Mean)/2;
        XPooled_sd = (sqrt(var(NeuralNetFeatures.Features)+0.01)+DataStats.Std)/2;
    end
else
    XPooled_mean=DataStats.Mean;
    XPooled_sd=DataStats.Std;
end


%subtract mean and divide by std
NeuralNetFeatures.Features = bsxfun(@rdivide, bsxfun(@minus, NeuralNetFeatures.Features, XPooled_mean), XPooled_sd);
if optionsNormalize.Binarize
    NeuralNetFeatures.Features(NeuralNetFeatures.Features>optionsNormalize.BinarizationThreshold)=1;
    NeuralNetFeatures.Features(NeuralNetFeatures.Features<optionsNormalize.BinarizationThreshold)=0;
end


% %normalize or binarize
% if optionsNormalize.Binarize
%     NeuralNetFeatures.Features(NeuralNetFeatures.Features>0)=1;
%     NeuralNetFeatures.Features(NeuralNetFeatures.Features<0)=0;
% else
%     %subtract mean and divide by std
%     NeuralNetFeatures.Features = bsxfun(@rdivide, bsxfun(@minus, NeuralNetFeatures.Features, XPooled_mean), XPooled_sd);
% end


%NORMALIZATION/BINARIZATION, END    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%also include the RF data and all options, for future reference
NeuralNetFeatures.optionsAll=options;
NeuralNetFeatures.optionsPrev=options;
NeuralNetFeatures.RFData=ReceptiveFields;
NeuralNetFeatures.Mean=XPooled_mean;
NeuralNetFeatures.Std=XPooled_sd;
NeuralNetFeatures.SparsifyFlag=optionsExtract.SparsifyFlag;
NeuralNetFeatures.SparsityMultiplier=optionsExtract.SparsityMultiplier;
NeuralNetFeatures.SummationType=optionsPool.SummationType;
NeuralNetFeatures.Binarize=optionsNormalize.Binarize;
NeuralNetFeatures.DatasetName=Dataset.DatasetName;

%save if instructed
if optionsNormalize.SaveFeatures
    if strcmp(optionsData.TrainTest,'Train')
        FilePrefix=strcat('FeaturesPooledNormalizedTrain_',int2str(optionsData.BatchNumber),'_');
    else
        FilePrefix=strcat('FeaturesPooledNormalizedTest_',int2str(optionsData.BatchNumber),'_');
    end
    save(fullfile(optionsNormalize.SavePath,strcat(FilePrefix,Dataset.DatasetName,'_NoRFs',int2str(numCentroids),'_RFSize',int2str(rfSize),'_Sp',int2str(int8(optionsExtract.SparsifyFlag)),...
        '_SMp',int2str(100*optionsExtract.SparsityMultiplier),'_',optionsPool.SummationType,'_Bin',int2str(int8(optionsNormalize.Binarize)))),'NeuralNetFeatures')
    fprintf('Features saved. \n');
end


%show the finalization on command window
fprintf('Feature Extraction, Pooling, Normalization done! % d:   \n', toc);

end%end function


