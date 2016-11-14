function NeuralNetFeatures=ExtractPoolNormalizeBinaryFeatures(Dataset, ReceptiveFields, DataStats, options)
%load options of every stage in a separate struct
optionsData=options{1};
optionsExtract=options{2};
optionsPool=options{3};
optionsNormalize=options{4};
NumberOfBits = options{8}.BitsPerRFs;

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

centroids=single(ReceptiveFields.RFs);
% cc = sum(centroids.^2, 2)'; %squares of centroids; we will be use it in distance computation; (x-y)^2=x^2+(y^2-2*x*y)==> cc=x
numCentroids = size(ReceptiveFields.RFs,1);
NumChannels=size(ReceptiveFields.RFs,2)/((ReceptiveFields.RFSize)^2);
rfSize=ReceptiveFields.RFSize;

%number of rows and columns in the feature space
prows = ImageDim(1)-rfSize+1;
pcols = ImageDim(2)-rfSize+1;

%measure computation time
tic


FeatureImage=zeros(optionsExtract.BatchSize,prows,pcols,numCentroids,'single');
patches=zeros(prows*pcols*optionsExtract.BatchSize,size(centroids,2),'single');
MeanRF=single(ReceptiveFields.M);
StdRF=single(ReceptiveFields.P);
z=zeros(size(patches,1),size(centroids,1));

if optionsExtract.GPU
%     X=gpuArray(single(X));
    centroids=gpuArray(centroids);
    FeatureImage=gpuArray(FeatureImage);
    MeanRF=gpuArray(MeanRF);
    StdRF=gpuArray(StdRF);
    z=gpuArray(z);
%     patches=gpuArray(single(patches));
end

if optionsData.Verbose 
    fprintf('Feature Extraction, Pooling, Normalization Start: \n'); 
end




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
            else      %Here, we take R,G,B components of RGB image separartely !!! it means we take an gray image!!!
                patches_ = [ im2col(reshape(X(j,1:OneImageSize),ImageDim(1:2)), [rfSize rfSize]) ;                   % R-component
                             im2col(reshape(X(j,OneImageSize+1:2*OneImageSize),ImageDim(1:2)), [rfSize rfSize]) ;    % G-component
                             im2col(reshape(X(j,2*OneImageSize+1:end),ImageDim(1:2)), [rfSize rfSize]) ]';           % B-componen
                         %%%%%%%%_____we extract patches(27) from images, with s=1; and rfSize=[6 6]_________%%%%%%%%%%            
            end
        end
                        OrgImageDim = Dataset.ImageDim;
                        rfSize = options{1,8}.rfSize; 
                        Dataset.ImageDim = [rfSize rfSize 3];

                        % Sampling the RFs with BRIEF descriptor is implemted here...
                        patches_binary = SamplingFrom_Spatial2(Dataset, patches_, options);
                        Dataset.ImageDim = OrgImageDim;

        patches((ImageNo-1)*size(patches_binary,1)+1:ImageNo*size(patches_binary,1),:)=patches_binary;
        ImageNo=ImageNo+1;
    end
    
    %if GPU processing put data on GPU
    if optionsExtract.GPU
        patches=gpuArray(patches);
    end
    
    % normalize for contrast
%     patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
% 
%     % whiten
%     patches = bsxfun(@minus, patches, MeanRF) * StdRF;
    
    
    for s=1:size(centroids,1)
%         z(s,:)=sum(bsxfun(@xor,patches(s,:), centroids),2);
        z(:,s) = sum(bsxfun(@xor,centroids(s,:), patches),2);
    end
    
    

    if optionsExtract.SparsifyFlag
        mu = mean(z, 2); % average distance to centroids for each patch
        patches2 = max(bsxfun(@minus, mu*optionsExtract.SparsityMultiplier, z), 0);
    else
        patches2=z;
    end % patches is now the data matrix of activations for each patch

    % reshape to numCentroids-channel image
    for kk=1:1:optionsExtract.BatchSize
        FeatureImage(kk,:,:,:)=reshape(patches2((kk-1)*size(patches_binary,1)+1:kk*size(patches_binary,1),:),prows, pcols, numCentroids);
    end

    
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

%binarize the neural activities (or make integer)
if optionsNormalize.Binarize
    if strcmp(optionsNormalize.BinarizeType,'Simple') %ones and zeros, the same size
        NeuralNetFeatures.Features(NeuralNetFeatures.Features>optionsNormalize.BinarizationThreshold)=1;
        NeuralNetFeatures.Features(NeuralNetFeatures.Features<optionsNormalize.BinarizationThreshold)=0;
    elseif strcmp(optionsNormalize.BinarizeType,'QuantizeIndicatorBinary') %indicator vectorize
        FeatTemp=NeuralNetFeatures.Features';
        FeatTemp=FeatTemp(:);
        FeatTemp = quantiz(FeatTemp,optionsNormalize.QuantizationThresholds)+1;
        FeatTemp=sparse(FeatTemp,1:length(FeatTemp),1);
        FeatTemp=full(FeatTemp)';
        FeatTemp=FeatTemp';
        FeatTemp=reshape(FeatTemp,[size(NeuralNetFeatures.Features,2)*(length(optionsNormalize.QuantizationThresholds)+1),size(NeuralNetFeatures.Features,1)]);
        FeatTemp=FeatTemp';
        NeuralNetFeatures.Features=FeatTemp;
    elseif strcmp(optionsNormalize.BinarizeType,'QuantizeInteger') %quantize and make integer
        FeatTemp=NeuralNetFeatures.Features';
        FeatTemp=FeatTemp(:);
        FeatTemp = quantiz(FeatTemp,optionsNormalize.QuantizationThresholds)+1;
        FeatTemp=reshape(FeatTemp,[size(NeuralNetFeatures.Features,2),size(NeuralNetFeatures.Features,1)]);
        FeatTemp=FeatTemp';
        NeuralNetFeatures.Features=FeatTemp;
    elseif strcmp(optionsNormalize.BinarizeType,'QuantizeIntegerBinary') %quantize, make integer then binarize
        FeatTemp=NeuralNetFeatures.Features';
        FeatTemp=FeatTemp(:);
        FeatTemp = quantiz(FeatTemp,optionsNormalize.QuantizationThresholds);
        FeatTemp=de2bi(FeatTemp,log2(length(optionsNormalize.QuantizationThresholds)+1));
        FeatTemp=FeatTemp';
        FeatTemp=reshape(FeatTemp,[size(NeuralNetFeatures.Features,2)*log2(length(optionsNormalize.QuantizationThresholds)+1),size(NeuralNetFeatures.Features,1)]);
        FeatTemp=FeatTemp';
        NeuralNetFeatures.Features=FeatTemp;
    else %k-means based centroid encoding.
        
        
    end
    
%     %convert zeros to minus one
%     NeuralNetFeatures.Features(NeuralNetFeatures.Features==0)=-1;

%     %whitening for debug purposes
%     C = cov(NeuralNetFeatures.Features);
%     M = mean(NeuralNetFeatures.Features);
%     [V,D] = eig(C);
%     P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
%     NeuralNetFeatures.Features = bsxfun(@minus, NeuralNetFeatures.Features, M) * P;
end

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
        FilePrefix=strcat('FeaturesPooledNormalized_Binary_Train_',int2str(optionsData.BatchNumber),'_');
    else
        FilePrefix=strcat('FeaturesPooledNormalized_Binary_Test_',int2str(optionsData.BatchNumber),'_');
    end
    save(fullfile(optionsNormalize.SavePath,strcat(FilePrefix,Dataset.DatasetName,'_',int2str(NumberOfBits),'Bits_NoRFs',int2str(numCentroids),'_RFSize',int2str(rfSize),'_Sp',int2str(int8(optionsExtract.SparsifyFlag)),...
        '_SMp',int2str(100*optionsExtract.SparsityMultiplier),'_',optionsPool.SummationType,'_Bin',int2str(int8(optionsNormalize.Binarize)))),'NeuralNetFeatures','-v7.3')
    if optionsData.Verbose 
        fprintf('Features saved. \n'); 
    end
end


%show the finalization on command window
if optionsData.Verbose 
    fprintf('Feature Extraction, Pooling, Normalization done! % d:   \n', toc); 
end


end