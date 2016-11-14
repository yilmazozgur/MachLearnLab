%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function pools the hidden layer activities of a single layer NN
% dimensionality reduction, to be used in computer vision tasks.
%
%%% Input:
%NeuralNetFeatures : A struct computed by ExtractSingleLayerFeature.m
%   it has the hidden layer activities of a NN. 
%options: struct for options.
%options.SummationType: The number of regions to be pooled is determined 
%   by this option. 4by4 'Fine', 2by2 'Coarse' or 1by1 'Single' reductions.
%   Default='coarse';
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
%Ref: It is based on Adam Coates code, Coates and Ng, 2011 paper.
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function PooledFeatures=PoolFeatures(NeuralNetFeatures,options)

%measure computation time
tic

%default options, if not given by the user
if nargin < 2
    options.SummationType='Coarse';
    options.GPU=false;
    options.SaveFeatures=false;
    options.SavePath='';
end

%extract the raw data and image dimensions
ReceptiveFields=NeuralNetFeatures.RFData;
ImageDim=ReceptiveFields.ImageDim;

%infer knowledge on the RFs
numCentroids = size(ReceptiveFields.RFs,1);
rfSize=ReceptiveFields.RFSize;

%Show what is to be done
fprintf('Pool Features Start: \n'); 

%process each image in the data one by one in a for loop
for i=1:size(NeuralNetFeatures.Features,2)
    NNFeature=NeuralNetFeatures.Features{i};
    
    % find rows and cols
    prows = ImageDim(1)-rfSize+1;
    pcols = ImageDim(2)-rfSize+1;

    % compute half sizes
    halfr = round(prows/2);
    halfc = round(pcols/2);

    %depending on the choice, get 4by4, 2by2 or 1by1 (whole) reductions
    if strcmp(options.SummationType,'Fine')
        %halfhalf sizes
        halfrhalf=round(halfr/2);
        halfchalf=round(halfc/2);
        q1_q1 = sum(sum(NNFeature(1:halfrhalf, 1:halfchalf, :), 1),2);
        q1_q2 = sum(sum(NNFeature(halfrhalf+1:halfr, 1:halfchalf, :), 1),2);
        q1_q3 = sum(sum(NNFeature(1:halfrhalf, halfchalf+1:halfc, :), 1),2);
        q1_q4 = sum(sum(NNFeature(halfrhalf+1:halfr, halfchalf+1:halfc, :), 1),2);
        q2_q1 = sum(sum(NNFeature(halfr+1:halfr+halfrhalf, 1:halfchalf, :), 1),2);
        q2_q2 = sum(sum(NNFeature(halfr+halfrhalf+1:end, 1:halfchalf, :), 1),2);
        q2_q3 = sum(sum(NNFeature(halfr+1:halfr+halfrhalf, halfchalf+1:halfc, :), 1),2);
        q2_q4 = sum(sum(NNFeature(halfr+halfrhalf+1:end, halfchalf+1:halfc, :), 1),2);
        q3_q1 = sum(sum(NNFeature(1:halfrhalf, halfc+1:halfc+halfchalf, :), 1),2);
        q3_q2 = sum(sum(NNFeature(halfrhalf+1:halfr, halfc+1:halfc+halfchalf, :), 1),2);
        q3_q3 = sum(sum(NNFeature(1:halfrhalf, halfc+halfchalf+1:end, :), 1),2);
        q3_q4 = sum(sum(NNFeature(halfrhalf+1:halfr, halfc+halfchalf+1:end, :), 1),2);
        q4_q1 = sum(sum(NNFeature(halfr+1:halfr+halfrhalf, halfc+1:halfc+halfchalf, :), 1),2);
        q4_q2 = sum(sum(NNFeature(halfr+halfrhalf+1:end, halfc+1:halfc+halfchalf, :), 1),2);
        q4_q3 = sum(sum(NNFeature(halfr+1:halfr+halfrhalf, halfc+halfchalf+1:end, :), 1),2);
        q4_q4 = sum(sum(NNFeature(halfr+halfrhalf+1:end, halfc+halfchalf+1:end, :), 1),2);
        XPooled = [q1_q1(:);q1_q2(:);q1_q3(:);q1_q4(:);q2_q1(:);q2_q2(:);q2_q3(:);q2_q4(:);q3_q1(:);q3_q2(:);q3_q3(:);q3_q4(:);q4_q1(:);q4_q2(:);q4_q3(:);q4_q4(:)]';
    elseif strcmp(options.SummationType,'Coarse')
        q1 = sum(sum(NNFeature(1:halfr, 1:halfc, :), 1),2);
        q2 = sum(sum(NNFeature(halfr+1:end, 1:halfc, :), 1),2);
        q3 = sum(sum(NNFeature(1:halfr, halfc+1:end, :), 1),2);
        q4 = sum(sum(NNFeature(halfr+1:end, halfc+1:end, :), 1),2);
        XPooled = [q1(:);q2(:);q3(:);q4(:)]';
    elseif strcmp(options.SummationType,'Single')
        q_sum = squeeze(sum(sum(NNFeature, 1),2));
        XPooled = [q_sum]';
    end

    PooledFeatures.Features{i}=XPooled;
    
end

%also include the RF data, for future reference
PooledFeatures.options=options;
PooledFeatures.RFData=ReceptiveFields;
optionsAll{1}=NeuralNetFeatures.options;
optionsAll{2}=NeuralNetFeatures.optionsPrev;
PooledFeatures.optionsPrev=optionsAll;

%bookkeeping of the choices
PooledFeatures.SparsifyFlag=NeuralNetFeatures.SparsifyFlag;
PooledFeatures.SparsityMultiplier=NeuralNetFeatures.SparsityMultiplier;
PooledFeatures.RatioNonZero=NeuralNetFeatures.RatioNonZero;
PooledFeatures.TrainOrTest=NeuralNetFeatures.TrainOrTest;
PooledFeatures.SummationType=options.SummationType;

%save if instructed
if options.SaveFeatures
    if strcmp(PooledFeatures.TrainOrTest,'Train')
        FilePrefix='PooledFeaturesTrain_';
    else
        FilePrefix='PooledFeaturesTest_';
    end
    save(fullfile(options.SavePath,strcat(FilePrefix,ReceptiveFields.DatasetName,'_NoRFs',int2str(numCentroids),'_RFSize',int2str(rfSize),'_Sp',int2str(int8(NeuralNetFeatures.SparsifyFlag)),...
        '_SMp',int2str(100*NeuralNetFeatures.SparsifyFlag),'_',options.SummationType)),'PooledFeatures')
    fprintf('Pooled Features saved. \n');
end

%show the finalization on command window
fprintf('Pool Features done!:   %d \n',toc);

end%end function


