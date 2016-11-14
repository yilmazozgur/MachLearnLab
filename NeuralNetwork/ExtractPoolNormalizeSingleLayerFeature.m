%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function computed single layer NN activities to be used 
%for vision/machine learning tasks.
%
%%% Input:
%Dataset : A struct of the dataset created by LoadDataset.m
%ReceptiveFields: A struct, that has centroids of RFs, mean and covariance 
%   matrices.
%options: struct for options. 
%options.SparsifyFlag: flag for sparsification of neural responses. 
%   Default=true.
%options.SparsityMultiplier: the amounf of sparsity. Default=0.7.
%options.GPU: enable GPU computing. Default=false.
%options.TrainOrTest: Train or Test data will be worked on. Default: 'Train'.
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
%Ref: It is based on Adam Coates code, Coates and Ng, 2011 paper.
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function NeuralNetFeatures=ExtractPoolNormalizeSingleLayerFeature(Dataset,ReceptiveFields,DataStats,options)

optionsExtract=options{1};
optionsPool=options{2};
optionsNormalize=options{3};

%default options, if not given by the user
if nargin < 4
    optionsExtract.SparsifyFlag=true;
    optionsExtract.SparsityMultiplier=0.7;
    optionsExtract.GPU=false;
    optionsExtract.TrainOrTest='Train';
    optionsExtract.SaveFeatures=false;
    optionsExtract.SavePath='';
    
    optionsPool.SummationType='Coarse';
    optionsPool.GPU=false;
    optionsPool.SaveFeatures=false;
    optionsPool.SavePath='';

    optionsNormalize.Binarize=true;
    optionsNormalize.GPU=false;
    optionsNormalize.SaveFeatures=false;
    optionsNormalize.SavePath='F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\SavedData';

end

%extract the raw data and image dimensions
if strcmp(optionsExtract.TrainOrTest,'Train') %either Train or Test data is processed.
    X=Dataset.trainX; %data
else
    X=Dataset.testX; %data
end
ImageDim=Dataset.ImageDim;
OneImageSize=ImageDim(1)*ImageDim(2);

%infer knowledge on the RFs
centroids=ReceptiveFields.RFs;
numCentroids = size(ReceptiveFields.RFs,1);
NumChannels=size(ReceptiveFields.RFs,2)/((ReceptiveFields.RFSize)^2);
rfSize=ReceptiveFields.RFSize;

%measure computation time
tic

%print what is to be done
fprintf('Feature Extraction, Pooling, Normalization start \n');

%process each image in the data one by one in a for loop
for i=1:size(X,1)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FIND HIDDEN LAYER ACTIVITIES, START

    % extract overlapping sub-patches into rows of 'patches'
    if ReceptiveFields.ImageIsGrayFlag
        patches = (im2col(reshape(X(i,1:OneImageSize),ImageDim(1:2)), [rfSize rfSize]))';
    else
        if ReceptiveFields.GrayConvertFlag
            patches = (sqrt(im2col(reshape(X(i,1:OneImageSize),ImageDim(1:2)), [rfSize rfSize]).^2+...
                im2col(reshape(X(i,OneImageSize+1:2*OneImageSize),ImageDim(1:2)), [rfSize rfSize]).^2 +...
                im2col(reshape(X(i,2*OneImageSize+1:end),ImageDim(1:2)), [rfSize rfSize]).^2) )';
        else      
            patches = [ im2col(reshape(X(i,1:OneImageSize),ImageDim(1:2)), [rfSize rfSize]) ;
                im2col(reshape(X(i,OneImageSize+1:2*OneImageSize),ImageDim(1:2)), [rfSize rfSize]) ;
                im2col(reshape(X(i,2*OneImageSize+1:end),ImageDim(1:2)), [rfSize rfSize]) ]';
        end
    end
    
    %if GPU processing put data on GPU
    if optionsExtract.GPU
        patches=gpuArray(single(patches));
        centroids=gpuArray(single(centroids));
    end
    
    % normalize for contrast
    patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

    % whiten
    patches = bsxfun(@minus, patches, ReceptiveFields.M) * ReceptiveFields.P;

    % compute 'triangle' activation function
    xx = sum(patches.^2, 2);
    cc = sum(centroids.^2, 2)';
    xc = patches * centroids';

    z = sqrt( bsxfun(@plus, cc, bsxfun(@minus, xx, 2*xc)) ); % distances


    if optionsExtract.SparsifyFlag
        mu = mean(z, 2); % average distance to centroids for each patch
        patches = max(bsxfun(@minus, mu*optionsExtract.SparsityMultiplier, z), 0);
    else
        patches=z;
    end % patches is now the data matrix of activations for each patch


    % reshape to numCentroids-channel image
    prows = ImageDim(1)-rfSize+1;
    pcols = ImageDim(2)-rfSize+1;
    FeatureImage=reshape(patches, prows, pcols, numCentroids);
    %gather data if GPU is on
    if optionsExtract.GPU
        FeatureImage=gather(FeatureImage);
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

    %depending on the choice, get 4by4, 2by2 or 1by1 (whole) reductions
    if strcmp(optionsPool.SummationType,'Fine')
        %halfhalf sizes
        halfrhalf=round(halfr/2);
        halfchalf=round(halfc/2);
        q1_q1 = sum(sum(FeatureImage(1:halfrhalf, 1:halfchalf, :), 1),2);
        q1_q2 = sum(sum(FeatureImage(halfrhalf+1:halfr, 1:halfchalf, :), 1),2);
        q1_q3 = sum(sum(FeatureImage(1:halfrhalf, halfchalf+1:halfc, :), 1),2);
        q1_q4 = sum(sum(FeatureImage(halfrhalf+1:halfr, halfchalf+1:halfc, :), 1),2);
        q2_q1 = sum(sum(FeatureImage(halfr+1:halfr+halfrhalf, 1:halfchalf, :), 1),2);
        q2_q2 = sum(sum(FeatureImage(halfr+halfrhalf+1:end, 1:halfchalf, :), 1),2);
        q2_q3 = sum(sum(FeatureImage(halfr+1:halfr+halfrhalf, halfchalf+1:halfc, :), 1),2);
        q2_q4 = sum(sum(FeatureImage(halfr+halfrhalf+1:end, halfchalf+1:halfc, :), 1),2);
        q3_q1 = sum(sum(FeatureImage(1:halfrhalf, halfc+1:halfc+halfchalf, :), 1),2);
        q3_q2 = sum(sum(FeatureImage(halfrhalf+1:halfr, halfc+1:halfc+halfchalf, :), 1),2);
        q3_q3 = sum(sum(FeatureImage(1:halfrhalf, halfc+halfchalf+1:end, :), 1),2);
        q3_q4 = sum(sum(FeatureImage(halfrhalf+1:halfr, halfc+halfchalf+1:end, :), 1),2);
        q4_q1 = sum(sum(FeatureImage(halfr+1:halfr+halfrhalf, halfc+1:halfc+halfchalf, :), 1),2);
        q4_q2 = sum(sum(FeatureImage(halfr+halfrhalf+1:end, halfc+1:halfc+halfchalf, :), 1),2);
        q4_q3 = sum(sum(FeatureImage(halfr+1:halfr+halfrhalf, halfc+halfchalf+1:end, :), 1),2);
        q4_q4 = sum(sum(FeatureImage(halfr+halfrhalf+1:end, halfc+halfchalf+1:end, :), 1),2);
        XPooled = [q1_q1(:);q1_q2(:);q1_q3(:);q1_q4(:);q2_q1(:);q2_q2(:);q2_q3(:);q2_q4(:);q3_q1(:);q3_q2(:);q3_q3(:);q3_q4(:);q4_q1(:);q4_q2(:);q4_q3(:);q4_q4(:)]';
    elseif strcmp(optionsPool.SummationType,'Coarse')
        q1 = sum(sum(FeatureImage(1:halfr, 1:halfc, :), 1),2);
        q2 = sum(sum(FeatureImage(halfr+1:end, 1:halfc, :), 1),2);
        q3 = sum(sum(FeatureImage(1:halfr, halfc+1:end, :), 1),2);
        q4 = sum(sum(FeatureImage(halfr+1:end, halfc+1:end, :), 1),2);
        XPooled = [q1(:);q2(:);q3(:);q4(:)]';
    elseif strcmp(optionsPool.SummationType,'Single')
        q_sum = squeeze(sum(sum(FeatureImage, 1),2));
        XPooled = [q_sum]';
    end

%POOL ACTIVITIES, END    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %save the feature for each image, in a list
    NeuralNetFeatures.Features(i,:) = XPooled;


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%NORMALIZATION/BINARIZATION, START   

%if it is training data, compute statistics,
%if not, use the stats provided as input for test data
if strcmp(optionsExtract.TrainOrTest,'Train')
    XPooled_mean = mean(NeuralNetFeatures.Features);
    XPooled_sd = sqrt(var(NeuralNetFeatures.Features)+0.01);
else
    XPooled_mean=DataStats.Mean;
    XPooled_sd=DataStats.Std;
end
%subtract mean and divide by std
NeuralNetFeatures.Features = bsxfun(@rdivide, bsxfun(@minus, NeuralNetFeatures.Features, XPooled_mean), XPooled_sd);

if optionsNormalize.Binarize
    NeuralNetFeatures.Features(NeuralNetFeatures.Features>0)=1;
    NeuralNetFeatures.Features(NeuralNetFeatures.Features<0)=0;
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

%save if instructed
if optionsNormalize.SaveFeatures
    if strcmp(optionsExtract.TrainOrTest,'Train')
        FilePrefix='FeaturesPooledNormalizedTrain_';
    else
        FilePrefix='FeaturesPooledNormalizedTest_';
    end
    save(fullfile(optionsNormalize.SavePath,strcat(FilePrefix,ReceptiveFields.DatasetName,'_NoRFs',int2str(numCentroids),'_RFSize',int2str(rfSize),'_Sp',int2str(int8(optionsExtract.SparsifyFlag)),...
        '_SMp',int2str(100*optionsExtract.SparsityMultiplier),'_',optionsPool.SummationType,'_Bin',int2str(int8(optionsNormalize.Binarize)))),'NeuralNetFeatures')
    fprintf('Features saved. \n');
end


%show the finalization on command window
fprintf('Feature Extraction, Pooling, Normalization done! % d:   \n', toc);

end%end function


