%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function performs experiments on specific computer vision dataset/task
%   step by step. MemoryEff refersd to bundling of steps 4,5 and 6 for RAM
%   effieciency, inside one function named:
%   LabExtractFeatures.
%
%%% Workflow:
% 1. Parameters are loaded in a separate function. 
% 2. Dataset is loaded using LoadDataset function. (Precomputed features automatically loaded via 'autoload')
% 3. Unsupervised learning is done on data. (eg. KmeansUnsupervised)
% 4. Features are extracted or neural network is applied on the dataset.
% 5. Extracted features are pooled. (or other reduction techniques)
% 6. Features are further expanded by other algorithms. (eg. Cellular Automata Reservoir)    
% 7. Supervised learing is performed using minFunc.
% 8. Performance metrics are calculated, depending on the dataset/task.
%
% Detail 1: Dataset can be handled in multiple batches. Ensemble learning
%   (both bagging and boosting) can be used in this context.
% Detail 2: Single layer activities can be binarized. (Must for CA
%   expansion). Several options available.
% Detail 3: There are many options in supervised learning: grid search,
%   early stopping, feature subset selection, RBF kernel.
% Detail 4: Autoload option is very useful for avoiding recalculation.
%   WhichData parameter determines the loaded files. 
% Detail 5: GPU computation is exploited in almost all steps. It speeds up
%   the computation significantly.  
%
%From:
%TOU_ML
%Ozgur Yilmaz, Turgut Ozal University, Ankara
%Web: ozguryilmazresearch.net
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;
%Load parameters and options for this experiment
ParametersClassification_Alisher_Kadir; % 'ParametersClassification', 'ParametersClassification_Alisher'

%put every option in a list, for monolithic feature computation in one
%function.
optionsAll{1}=optionsData;
optionsAll{2}=optionsExtractFeatures;
optionsAll{3}=optionsPoolFeatures;
optionsAll{4}=optionsNormalizeBinarize;
optionsAll{5}=optionsExpansion;
optionsAll{6}=optionsSupervisedLearning;
optionsAll{7}=optionsEvaluation;
optionsAll{8}=optionsKmeans;
optionsAll{9}=optionsBriefFeatureExtraction;
optionsAll{10}=RootPath;
optionsAll_Org=optionsAll;


%Run unsupervised learning
OrgLoadDataOption=optionsAll{1}.WhichData;
optionsAll{1}.WhichData='Raw';
%Load dataset, raw for unsupervised pre-training
[DatasetRaw, optionsAll]=LoadDataset_Kadir(DatasetName,RootPath,optionsAll);
optionsAll{1}.WhichData=OrgLoadDataOption; %restore original WhichData option
ReceptiveFields=KmeansUnsupervised_Kadir(DatasetRaw,NumberOfRFs,rfSize,optionsAll);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TRAIN, START

%TRAIN initially with the first batch
%Extract single layer neural net features
optionsAll{1}.TrainTest='Train';
%Load dataset, train
[Dataset, optionsAll]=LoadDataset_Kadir(DatasetName,RootPath,optionsAll);

NeuralNetFeatures_PooledNormalized_Train=LabExtractFeatures(Dataset,ReceptiveFields,[],optionsAll);
%Expand the space with cellular automata features, or polynomial/kernel
%features etc. see minFunc_examples.m
NeuralNetFeatures_PooledNormalized_Train=ExpandSpace_v3(NeuralNetFeatures_PooledNormalized_Train,[],optionsAll);

%put data stats into a struct for further processing with more batches
%form data stats, for normalization. it will be combined with the
%current batch inside the function (see normalization part). 
DataStats.Mean=NeuralNetFeatures_PooledNormalized_Train.Mean;
DataStats.Std=NeuralNetFeatures_PooledNormalized_Train.Std;
DataStats.ExMean=NeuralNetFeatures_PooledNormalized_Train.ExMean;
DataStats.ExStd=NeuralNetFeatures_PooledNormalized_Train.ExStd;
DataStats.RandomPermutation=NeuralNetFeatures_PooledNormalized_Train.RandomPermutation;
DataStats.SamplingMask=NeuralNetFeatures_PooledNormalized_Train.SamplingMask;

%Supervised learning
Model=SupervisedLearning_Kadir([],NeuralNetFeatures_PooledNormalized_Train,optionsAll);
%Save models of each batch separately, for possible ensemble classification
Model_Batches{1,1}=Model; %First Batch, First Epoch
ValidationAccuracyEpochs(1,1)=Model.ValidationAccuracy;
ValidationBPCEpochs(1,1)=Model.Validation_BPC;



%record these for test data computation
OriginalWhichData=optionsAll{1}.WhichData; 
OriginalAutoLoad=optionsAll{1}.AutoLoad;

%repeat training for many epochs
NumberOfEpochs=optionsSupervisedLearning.Epochs;
CrossValidationAccuracyEpoch_Prev=-1;
MaxCrossValidationAccuracy=0;
for epochNo=1:1:NumberOfEpochs
    %keep training using the rest of the data batches
    for i=1:1:Dataset.NumberOfBatches
        fprintf('_____________________________________Epoch = %d/%d ; Batch = %d/%d_____________________________________\n',epochNo,NumberOfEpochs,i,Dataset.NumberOfBatches);
        if i==1 && epochNo==1 %in first epoch, skip first batch, because it was already processed above
            continue;
        end
        %Extract single layer neural net features
        optionsAll{1}.TrainTest='Train';
        optionsAll{1}.BatchNumber=i;
        %Load dataset, train
        [Dataset, optionsAll]=LoadDataset_Kadir(DatasetName,RootPath,optionsAll);
        
        NeuralNetFeatures_PooledNormalized_Train=LabExtractFeatures(Dataset,ReceptiveFields,DataStats,optionsAll);     
        %Expand the space with cellular automata features, or polynomial/kernel
        %features etc. see minFunc_examples.m
        NeuralNetFeatures_PooledNormalized_Train=ExpandSpace_v3(NeuralNetFeatures_PooledNormalized_Train,DataStats,optionsAll);

        %form data stats, for normalization. it will be combined with the
        %current batch inside the function (see normalization part of feature extraction function). 
        DataStats.Mean=NeuralNetFeatures_PooledNormalized_Train.Mean;
        DataStats.Std=NeuralNetFeatures_PooledNormalized_Train.Std;
        DataStats.ExMean=NeuralNetFeatures_PooledNormalized_Train.ExMean;
        DataStats.ExStd=NeuralNetFeatures_PooledNormalized_Train.ExStd;

        %if ensemble learning is chosen, then the models can be independent
        %from each other.
        if optionsAll{7}.Ensemble && optionsAll{7}.EnsembleModelIndependence
            Model=[];
        end

        %Supervised learning
        Model=SupervisedLearning_Kadir(Model,NeuralNetFeatures_PooledNormalized_Train,optionsAll);
        %Save models of each batch separately, for possible ensemble
        %classification
        if not(optionsSupervisedLearning.Greedy)
            Model_Batches{i,epochNo}=Model;
        elseif i==Dataset.NumberOfBatches
            Model_Batches{epochNo}=Model;
        end
        ValidationAccuracyEpochs(i,epochNo)=Model.ValidationAccuracy;
        ValidationBPCEpochs(i,epochNo)=Model.Validation_BPC;
        
        TrainAccuracyEpoch(i,epochNo)=Model.TrainAccuracy;
    end
    
    %report epoch accuracy if verbose
    if optionsData.Verbose
        fprintf('Epoch Validation Accuracy:  %d\n',mean(ValidationAccuracyEpochs(:,epochNo)))
        fprintf('Epoch Validation BPC:  %d\n',mean(ValidationBPCEpochs(:,epochNo)))
    end
    
    %stop training by checking cross validation error
    if optionsSupervisedLearning.EpochCrossValidation
        CrossValidationAccuracyEpoch=mean(ValidationAccuracyEpochs(:,epochNo));
        if CrossValidationAccuracyEpoch-CrossValidationAccuracyEpoch_Prev<optionsSupervisedLearning.EpochCrossValidationThreshold
            NumberOfEpochs=epochNo-1;
            fprintf('******************************Cross Validation in Epochs stopped training************************************: \n')
        end
        CrossValidationAccuracyEpoch_Prev=CrossValidationAccuracyEpoch;
    end
    
    %once the first epoch is executed, use the saved features.
    if epochNo==1 && optionsExpansion.Perform
        optionsAll{1}.AutoLoad=true;
        optionsAll{1}.WhichData='ExpandedFeatures';
    elseif epochNo==1 && not(optionsExpansion.Perform)
        optionsAll{1}.AutoLoad=true;
        optionsAll{1}.WhichData='Features';
    end
end
% analyze cross validation error and output the best intermediate model.
if optionsSupervisedLearning.EpochIterationDiagnostics
    MeanCVAccuracy=mean(ValidationAccuracyEpochs);
    %find max accuracy epoch, use that epoch's model as the only model.
    [MaxAcc, MaxAccIndex]=max(MeanCVAccuracy);
  
    if not(optionsSupervisedLearning.Greedy)
        Model_Batches_{size(Model_Batches,1)}=[]; % initialization of Model_Batches_ structure (10x1 cell) or (5x1 cell), for 10 batches dataset, and 5 batches dataset, respectively
        for ii=1:1:size(Model_Batches,1)
            Model_Batches_{ii}=Model_Batches{ii,MaxAccIndex};
        end
    else
        Model_Batches_=Model_Batches{MaxAccIndex}; % Save model (only) at Epoch that has the maximum CV Accuracy
    end
    Model_Batches=Model_Batches_;
end

%TRAIN, END
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TEST, START
%Measure performance on test data


%reset to original settings (possibly altered after several epochs)
optionsAll{1}.WhichData=OriginalWhichData;
optionsAll{1}.AutoLoad=OriginalAutoLoad;

optionsAll{1}.TrainTest='Test';
optionsAll{1}.BatchNumber=1;    %1
%Load dataset for test
[Dataset, optionsAll]=LoadDataset_Kadir(DatasetName,RootPath,optionsAll);

%extract features, test
NeuralNetFeatures_PooledNormalized_Test=LabExtractFeatures(Dataset,ReceptiveFields,DataStats,optionsAll);


%expand features, test
NeuralNetFeatures_PooledNormalized_Test=ExpandSpace_v3(NeuralNetFeatures_PooledNormalized_Test,DataStats,optionsAll);


Performance{1,Dataset.NumberOfBatches}=[]; % initialization of Performance structure
%compute test error
Performance{1}=EvaluateSupervisedLearning(Model_Batches,NeuralNetFeatures_PooledNormalized_Test,optionsAll);

%test with the rest of the batches
for i=2:1:Dataset.NumberOfBatches
    %extract features for test data
    optionsAll{1}.TrainTest='Test';
    optionsAll{1}.BatchNumber=i;
    %Load dataset for test
    [Dataset, optionsAll]=LoadDataset_Kadir(DatasetName,RootPath,optionsAll);

    %extract features, test
    NeuralNetFeatures_PooledNormalized_Test=LabExtractFeatures(Dataset,ReceptiveFields,DataStats,optionsAll);

    %expand features, test
    NeuralNetFeatures_PooledNormalized_Test=ExpandSpace_v3(NeuralNetFeatures_PooledNormalized_Test,DataStats,optionsAll);

    %compute test error
    Performance{i}=EvaluateSupervisedLearning(Model_Batches,NeuralNetFeatures_PooledNormalized_Test,optionsAll);
end

TrainAccuracy(1:size(Performance,2),1)=0;
TestAccuracy (1:size(Performance,2),2)=0;

%Average performance for batches
for i=1:1:size(Performance,2)
    TrainAccuracy(i,:)=Performance{1,i}.TrainAccuracy;
    TestAccuracy(i,:)=Performance{1,i}.TestAccuracy;
end
fprintf('Average Train Accuracy:  %f\n',mean(TrainAccuracy))
fprintf('Average Test Accuracy One Decision:  %f\n',mean(TestAccuracy(:,1)))
fprintf('Average Test Accuracy Two Decisions:  %f\n',mean(TestAccuracy(:,2)))

%TEST,END
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAVING EXPERIMENT RESULTS INTO TXT FILES
path_cell=strsplit(RootPath,'/');
SavePath=strcat('/',fullfile(path_cell{1,1:end-1}));
SavePath=strcat(SavePath,'/EXPERIMENT RESULTS');

switch optionsAll{2}.Type
    case 'ExtractBRIEF'
        switch optionsAll{9}.Type
            case 'BriefOverlappedPatches'
                FigureName=strcat('/home/comp1/Desktop/Recurrent Holistic Vision v0.4/MatlabCode/BRIEF/BriefOverlappedPatches/MeanCrossValidationAccuracies/',optionsAll{9}.Dimension,';MeanCVaccuracy',int2str(optionsAll{6}.Epochs),';',int2str(optionsAll{6}.MaxIter),';',optionsAll{9}.ColorType,';[',int2str(optionsAll{9}.PatchSize(1)),'x',int2str(optionsAll{9}.PatchSize(1)),']&',int2str(optionsAll{9}.Stride),';[',int2str(optionsAll{9}.NumberOfTests(1)),',',int2str(optionsAll{9}.NumberOfTests(2)),',',int2str(optionsAll{9}.NumberOfTests(3)),'];',int2str(optionsAll{9}.SmoothingFlag),';',num2str(optionsAll{6}.lambda),'.fig');
                h=figure;
                plot(MeanCVAccuracy);
                savefig(h,FigureName);
%                 close (h);
                switch optionsAll{9}.Dimension
                    case '3D'
                        fid = fopen(strcat(SavePath,'/3D-Brief Overlapped Patches.txt'),'a+');
                        fprintf(fid,'\t%d\t\t  %d\t\t\t  %s\t\t    [%d %d]\t\t  %d\t\t\t [%d]\t\t\t  %d;[%d,%d];%.3f\t\t\t  %d\t\t\t %.3f\t\t\t  %d\t\t\t  %d\t\t    %.3f\t\t %.3f\t\t %.3f - %d Batches\n\n',...
                               optionsAll{6}.Epochs,...
                               optionsAll{6}.MaxIter,...
                               optionsAll{9}.ColorType,...
                               optionsAll{9}.PatchSize(1),...
                               optionsAll{9}.PatchSize(2),...
                               optionsAll{9}.Stride,...
                               optionsAll{9}.NumberOfTests(1),...
                               optionsAll{9}.SmoothingFlag,...
                               optionsAll{9}.SmoothKernel(1),...
                               optionsAll{9}.SmoothKernel(2),...
                               optionsAll{9}.SD_SmoothingFilter,...      
                               optionsAll{6}.EpochCrossValidationThreshold,...
                               optionsAll{6}.lambda,...
                               optionsAll{6}.EpochIterationDiagnostics,...
                               optionsAll{6}.EarlyStopping,...
                               mean(TrainAccuracy),...
                               mean(TestAccuracy(:,1)),...
                               mean(TestAccuracy(:,2)),...
                               optionsAll{1}.TotalNumberOfBatches);
                        fclose(fid);
                    case '2D'
                        fid = fopen(strcat(SavePath,'/2D-Brief Overlapped Patches.txt'),'a+');
                        fprintf(fid,'\t%d\t\t  %d\t\t\t  %s\t\t    [%d %d]\t\t  %d\t\t [%d %d %d]\t\t  %d;[%d,%d];%.3f\t\t\t %d\t\t\t %.3f\t\t\t%d\t\t    %d\t\t    %.3f\t\t %.3f\t\t %.3f - %d Batches\n\n',...
                               optionsAll{6}.Epochs,...
                               optionsAll{6}.MaxIter,...
                               optionsAll{9}.ColorType,...
                               optionsAll{9}.PatchSize(1),...
                               optionsAll{9}.PatchSize(2),...
                               optionsAll{9}.Stride,...
                               optionsAll{9}.NumberOfTests(1),...
                               optionsAll{9}.NumberOfTests(2),...
                               optionsAll{9}.NumberOfTests(3),...
                               optionsAll{9}.SmoothingFlag,...
                               optionsAll{9}.SmoothKernel(1),...
                               optionsAll{9}.SmoothKernel(2),...
                               optionsAll{9}.SD_SmoothingFilter,...      
                               optionsAll{6}.EpochCrossValidationThreshold,...
                               optionsAll{6}.lambda,...
                               optionsAll{6}.EpochIterationDiagnostics,...
                               optionsAll{6}.EarlyStopping,...
                               mean(TrainAccuracy),...
                               mean(TestAccuracy(:,1)),...
                               mean(TestAccuracy(:,2)),...
                           optionsAll{1}.TotalNumberOfBatches);
                        fclose(fid);
                end


            case 'BriefSubImages'
                FigureName=strcat('/home/comp1/Desktop/Recurrent Holistic Vision v0.4/MatlabCode/BRIEF/BriefSubImages/MeanCrossValidationAccuracies/','MeanCVaccuracy',int2str(optionsAll{6}.Epochs),';',int2str(optionsAll{6}.MaxIter),';',optionsAll{9}.ColorType,';',optionsAll{3}.SummationType,';[',int2str(optionsAll{9}.NumberOfTests(1)),',',int2str(optionsAll{9}.NumberOfTests(2)),',',int2str(optionsAll{9}.NumberOfTests(3)),'];',int2str(optionsAll{9}.SmoothingFlag),';',num2str(optionsAll{6}.lambda),'.fig');
                h=figure;
                plot(MeanCVAccuracy);
                savefig(h,FigureName);
                close (h);

                switch optionsAll{9}.Dimension
                    case '3D'
                        fid = fopen(strcat(SavePath,'/3D-Brief SubImages.txt'),'a+');
                        fprintf(fid,' \t%d\t\t  %d\t\t\t  %s\t\t    %s\t\t\t [%d]\t\t\t %d;[%d,%d];%.3f\t\t\t %d\t\t %.3f\t\t\t  %d\t\t\t  %d\t\t      %0.3f\t\t   %.3f\t\t  %.3f - %d Batches\n\n',...
                               optionsAll{6}.Epochs,...
                               optionsAll{6}.MaxIter,...
                               optionsAll{9}.ColorType,...
                               optionsAll{3}.SummationType,...
                               optionsAll{9}.NumberOfTests(1),...
                               optionsAll{9}.SmoothingFlag,...
                               optionsAll{9}.SmoothKernel(1),...
                               optionsAll{9}.SmoothKernel(2),...
                               optionsAll{9}.SD_SmoothingFilter,...      
                               optionsAll{6}.EpochCrossValidationThreshold,...
                               optionsAll{6}.lambda,...
                               optionsAll{6}.EpochIterationDiagnostics,...
                               optionsAll{6}.EarlyStopping,...
                               mean(TrainAccuracy),...
                               mean(TestAccuracy(:,1)),...
                               mean(TestAccuracy(:,2)),...
                               optionsAll{1}.TotalNumberOfBatches);
                        fclose(fid);
                    case '2D'
                        fid = fopen(strcat(SavePath,'/2D-Brief SubImages.txt'),'a+');
                        fprintf(fid,'\t%d\t\t  %d\t\t\t  %s\t\t    %s\t   [%d %d %d]\t\t\t %d;[%d,%d];%.3f\t\t    %d\t\t   %.3f\t\t %d\t\t\t  %d\t\t    %.3f\t\t %.3f\t\t %.3f - %d Batches\n\n',...
                               optionsAll{6}.Epochs,...
                               optionsAll{6}.MaxIter,...
                               optionsAll{9}.ColorType,...
                               optionsAll{3}.SummationType,...
                               optionsAll{9}.NumberOfTests(1),...
                               optionsAll{9}.NumberOfTests(2),...
                               optionsAll{9}.NumberOfTests(3),...
                               optionsAll{9}.SmoothingFlag,...
                               optionsAll{9}.SmoothKernel(1),...
                               optionsAll{9}.SmoothKernel(2),...
                               optionsAll{9}.SD_SmoothingFilter,...      
                               optionsAll{6}.EpochCrossValidationThreshold,...
                               optionsAll{6}.lambda,...
                               optionsAll{6}.EpochIterationDiagnostics,...
                               optionsAll{6}.EarlyStopping,...
                               mean(TrainAccuracy),...
                               mean(TestAccuracy(:,1)),...
                               mean(TestAccuracy(:,2)),...
                               optionsAll{1}.TotalNumberOfBatches);
                        fclose(fid);
                end
        end
    case 'ExtractPoolNormalizeSingleLayerFeature_v3'
end     
% SAVING, END
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



