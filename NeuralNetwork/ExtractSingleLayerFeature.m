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


function NeuralNetFeatures=ExtractSingleLayerFeature(Dataset,ReceptiveFields,options)

%measure computation time
tic

%default options, if not given by the user
if nargin < 3
    options.SparsifyFlag=true;
    options.SparsityMultiplier=0.7;
    options.GPU=false;
    options.TrainOrTest='Train';
    options.SaveFeatures=false;
    options.SavePath='';
end

%extract the raw data and image dimensions
if strcmp(options.TrainOrTest,'Train') %either Train or Test data is processed.
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

%Show what is to be done
fprintf('Feature Extraction Start: \n'); 

%process each image in the data one by one in a for loop
for i=1:size(X,1)

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
    if options.GPU
        patches=gpuArray(patches);
        centroids=gpuArray(centroids);
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


    if options.SparsifyFlag
        mu = mean(z, 2); % average distance to centroids for each patch
        patches = max(bsxfun(@minus, mu*options.SparsityMultiplier, z), 0);
    else
        patches=z;
    end % patches is now the data matrix of activations for each patch


    % reshape to numCentroids-channel image
    prows = ImageDim(1)-rfSize+1;
    pcols = ImageDim(2)-rfSize+1;
    FeatureImage=reshape(patches, prows, pcols, numCentroids);
    %gather data if GPU is on
    if options.GPU
        FeatureImage=gather(FeatureImage);
    end
    NeuralNetFeatures.Features{i} = FeatureImage;


end

%also include the RF data, for future reference
NeuralNetFeatures.options=options;
NeuralNetFeatures.RFData=ReceptiveFields;
optionsAll{1}=ReceptiveFields.options;
optionsAll{2}=Dataset.options;
NeuralNetFeatures.optionsPrev=optionsAll;

%bookkeeping of the choices
NeuralNetFeatures.SparsifyFlag=options.SparsifyFlag;
NeuralNetFeatures.SparsityMultiplier=options.SparsityMultiplier;
NeuralNetFeatures.RatioNonZero=sum(NeuralNetFeatures.Features{1}(:)>0)/length(NeuralNetFeatures.Features{1}(:));
NeuralNetFeatures.TrainOrTest=options.TrainOrTest;

%save if instructed
if options.SaveFeatures
    if strcmp(NeuralNetFeatures.TrainOrTest,'Train')
        FilePrefix='FeaturesTrain_';
    else
        FilePrefix='FeaturesTest_';
    end
    save(fullfile(options.SavePath,strcat(FilePrefix,ReceptiveFields.DatasetName,'_NoRFs',int2str(numCentroids),'_RFSize',int2str(rfSize),'_Sp',int2str(int8(options.SparsifyFlag)),...
        '_SMp',int2str(100*options.SparsityMultiplier))),'NeuralNetFeatures')
    fprintf('Features saved. \n');
end


%show the finalization on command window
fprintf('Feature Extraction done!:   %d \n',toc);

end%end function


