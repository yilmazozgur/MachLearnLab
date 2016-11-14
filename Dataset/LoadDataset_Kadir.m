%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function loads a wide variety of datasets, generally for computer
%vision/machine learning tasks.
%Training and test data are separately saved into the Dataset struct.
%The structure of the data is Number of Instances * Number of Dimensions 
%for data matrix, and Number of Instances * Number of Predicted Vars for
%label matrix. 
%If there is a separate cross validation data, it is appended at the end of
%the training data. 
%
%%% Input:
%Name : A string of the dataset name, to be used in the switch-case
%RootPath: A string of the root path for the main codebase. Default: is
%   'F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\MatlabCode'
%options: struct for options. 
%optionsData.AutoLoad: automatically checks for the precomptued features
%   and model, loads them. Default=false;
%options.WhichData: Load raw data, processed data, or the model only. 
%   Default='Raw';
%options.TrainTest:Load train, test or all data. Default='Train';
%options.Randomize: randomize the order of dataset instances
%options.TrainDataRatio: The ratio of the train data to be used. For faster
%   training or experimental purposes. In [0,1] range.
%options.BatchNumber: For very large datasets, each batch can be accessed via
%   this function one by one, using the BatchNumber. Default=0 (no batch)
%
%%% Output:
%Dataset: A struct, that has train/test data matrices and
%label matrices. See below for details.
%
%
%From:
%TOU_ML
%Ozgur Yilmaz, Turgut Ozal University, Ankara
%Web: ozguryilmazresearch.net
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Dataset, options]=LoadDataset(Name,RootPath,options)

%load options of every stage in a separate struct
optionsData=options{1};
optionsExtract=options{2};
optionsPool=options{3};
optionsNormalize=options{4};
optionsExpansion=options{5};
optionsSupervisedLearning=options{6};


%default options, if not given by the user
if nargin < 3
    optionsData.AutoLoad=false;
    optionsData.WhichData='Raw'; %'Raw', 'Features', 'ExpandedFeatures', 'Model'
    optionsData.TrainTest='Train'; % 'Train', 'Test'
    optionsData.Randomize=false;
    optionsData.TrainDataRatio=1;
    optionsData.BatchNumber=0;
end

%load dataset paths, saved in a global mat file
load(fullfile(RootPath,'Dataset/DatasetPaths'));

%load raw datasets from the corresponding mat files for each dataset
%when batch number is zero, it means whole dataset is loaded at
%once
switch Name
    case 'CIFAR10'
        FileName=strcat(CIFAR10_DataPath,'_Batch',int2str(optionsData.BatchNumber),'.mat');
        load(FileName);
        ImageDim=[32 32 3];
        NumberOfBatches=optionsData.TotalNumberOfBatches;
        options{1}.NumberOfBatches=NumberOfBatches;
        options{1}.BinaryDataset=false;
        options{1}.DatasetSequenceFlag=false;
        
    case 'CIFAR100'
        load(CIFAR100_DataPath);  
        ImageDim=[32 32 3];
        NumberOfBatches=1;
        options{1}.BinaryDataset=false;
        options{1}.DatasetSequenceFlag=false;
        
    case 'KITTI'
        fileName   = strcat('KITTI', '_Batch', int2str(optionsData.BatchNumber), '.mat');
                                    
        load(fileName);
        ImageDim = [21 21 2];
        NumberOfBatches = optionsData.TotalNumberOfBatches;
        options{1}.NumberOfBatches=NumberOfBatches;
        options{1}.BinaryDataset=false;
        options{1}.DatasetSequenceFlag=false;
        
    case 'STL10'
    
    case 'SUN_Scene'
        
    case 'SUN_SceneAttributes'    
     
    case 'SUN_Objects'
        
    case 'MSCOCO_Objects'
        
    case 'Texture'
    
    case 'Material'
        
    case 'PennTreebank_Character'
        FileName=strcat(PennTreebank_Character_DataPath,'_Batch',int2str(optionsData.BatchNumber),'.mat');
        load(FileName);
        ImageDim=[32 32 3]; %not meaningful
        NumberOfBatches=100;
        options{1}.NumberOfBatches=NumberOfBatches;
        options{1}.BinaryDataset=true;
        options{1}.DatasetSequenceFlag=true;
    
end

%randomize data if flag is raised in options
if optionsData.Randomize
    %train data
    RandIndex=randperm(size(trainX,1));
    trainX=trainX(RandIndex,:);
    trainY=trainY(RandIndex);
    
    %test data
    RandIndex=randperm(size(testX,1));
    testX=testX(RandIndex,:);
    testY=testY(RandIndex);
end

%choose the subset (from beginning) of train data, if it is given in the
%options. 
if optionsData.TrainDataRatio<1
    trainY=trainY(1:round(optionsData.TrainDataRatio*size(trainX,1)));
    trainX=trainX(1:round(optionsData.TrainDataRatio*size(trainX,1)),:);
    
    %!!!!!for debugging purposes, remove LATER!!!!
    testY=testY(1:round(optionsData.TrainDataRatio*size(testY,1)));
    testX=testX(1:round(optionsData.TrainDataRatio*size(testX,1)),:);

end

%form the struct with auxillary data and options for future reference.
Dataset.options=options;

%load the data, for which the type is determined by options.WhichData
if strcmp(optionsData.WhichData,'Raw') %raw data from dataset.
    Dataset.trainX=trainX;
    Dataset.trainY=trainY;
    Dataset.testX=testX;
    Dataset.testY=testY;
    if optionsData.Verbose 
        fprintf('Raw Dataset Loaded \n'); 
    end
elseif strcmp(optionsData.WhichData,'Features') %precomputed features  
    %load the corresponding data depending on the choice:
    %optionsData.TrainTest, i.e. train or test data
    if strcmp(options{2}.Type,'ExtractBRIEF')
            if strcmp(options{9}.Type,'BriefOverlappedPatches')
                        savedir=strcat(RootPath,'/BRIEF/BriefOverlappedPatches/Saved BRIEF_OverlappedPatches Features/');
                        if strcmp(optionsData.TrainTest,'Train') %optionsData.TrainTest='Train'
                                if options{9}.SmoothingFlag
                                    if strcmp(options{9}.Dimension,'3D')
                                        FileName=strcat(savedir,'Train-',int2str(options{1}.TotalNumberOfBatches),';',options{1}.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_[',int2str(options{9}.PatchSize),'];',int2str(options{9}.Stride),'_{',int2str(options{9}.NumberOfTests(1)),'}_[',int2str(options{9}.SmoothKernel),']_',num2str(options{9}.SD_SmoothingFilter),'_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                    elseif strcmp(options{9}.Dimension,'2D')
                                        FileName=strcat(savedir,'Train-',int2str(options{1}.TotalNumberOfBatches),';',options{1}.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_[',int2str(options{9}.PatchSize),'];',int2str(options{9}.Stride),'_{',int2str(options{9}.NumberOfTests(1)),'}_[',int2str(options{9}.SmoothKernel),']_',num2str(options{9}.SD_SmoothingFilter),'_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                    end
                                else
                                    if strcmp(options{9}.Dimension,'3D')
                                        FileName=strcat(savedir,'Train-',int2str(options{1}.TotalNumberOfBatches),';',options{1}.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_[',int2str(options{9}.PatchSize),'];',int2str(options{9}.Stride),'_{',int2str(options{9}.NumberOfTests(1)),'}_[NO Smoothing]_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                    elseif strcmp(options{9}.Dimension,'2D') 
                                        FileName=strcat(savedir,'Train-',int2str(options{1}.TotalNumberOfBatches),';',options{1}.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_[',int2str(options{9}.PatchSize),'];',int2str(options{9}.Stride),'_{',int2str(options{9}.NumberOfTests),'}_[NO Smoothing]_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                    end
                                end
                            %autoload function. If file does not exist, decrement the stage (Features to Raw, Expanded Feature to Features etc) and try loading again.
                                if exist(FileName, 'file')
                                        load(FileName)        
                                        if optionsData.Verbose 
                                            disp('Features Loaded Train (No computation)') 
                                        end
                                        Dataset=Features;
                                elseif not(exist(FileName, 'file')) && optionsData.AutoLoad 
                                        options{1}.WhichData='Raw';
                                        [Dataset, options]=LoadDataset(Name,RootPath,options);
                                        disp('Warning: Intended precomputed feature file does not exist')
                                else
                                        msg = 'Loading error occured.';
                                        error(msg)
                                end
                        else %optionsData.TrainTest='Test';

                                if options{9}.SmoothingFlag
                                    if strcmp(options{9}.Dimension,'3D')
                                        FileName=strcat(savedir,'Test-',int2str(options{1}.TotalNumberOfBatches),';',options{1}.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_[',int2str(options{9}.PatchSize),'];',int2str(options{9}.Stride),'_{',int2str(options{9}.NumberOfTests(1)),'}_[',int2str(options{9}.SmoothKernel),']_',num2str(options{9}.SD_SmoothingFilter),'_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                    elseif strcmp(options{9}.Dimension,'2D')
                                        FileName=strcat(savedir,'Test-',int2str(options{1}.TotalNumberOfBatches),';',options{1}.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_[',int2str(options{9}.PatchSize),'];',int2str(options{9}.Stride),'_{',int2str(options{9}.NumberOfTests(1)),'}_[',int2str(options{9}.SmoothKernel),']_',num2str(options{9}.SD_SmoothingFilter),'_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                    end
                                else
                                    if strcmp(options{9}.Dimension,'3D')
                                        FileName=strcat(savedir,'Test-',int2str(options{1}.TotalNumberOfBatches),';',options{1}.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_[',int2str(options{9}.PatchSize),'];',int2str(options{9}.Stride),'_{',int2str(options{9}.NumberOfTests(1)),'}_[NO Smoothing]_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                    elseif strcmp(options{9}.Dimension,'2D') 
                                        FileName=strcat(savedir,'Test-',int2str(options{1}.TotalNumberOfBatches),';',options{1}.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_[',int2str(options{9}.PatchSize),'];',int2str(options{9}.Stride),'_{',int2str(options{9}.NumberOfTests),'}_[NO Smoothing]_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                    end
                                end
                                
                                    load(FileName)        
                                    if optionsData.Verbose 
                                        disp('Features Loaded Test (No computation)') 
                                    end
                                    Dataset=Features;
                        end
            elseif strcmp(options{9}.Type,'BriefSubImages')
                    savedir=strcat(RootPath,'/BRIEF/BriefSubImages/Saved BRIEF_SubImages Features');
                    if strcmp(optionsData.TrainTest,'Train')
                                        if options{9}.SmoothingFlag
                                            if strcmp(options{9}.Dimension,'3D')
                                                FileName=strcat(savedir,'Train-',int2str(options{1}.TotalNumberOfBatches),';',Dataset.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_',options{3}.SummationType,'_{',int2str(options{9}.NumberOfTests(1)),'}_[',int2str(options{9}.SmoothKernel),']_',num2str(options{9}.SD_SmoothingFilter),'_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                            elseif strcmp(options{9}.Dimension,'2D')
                                                FileName=strcat(savedir,'Train-',int2str(options{1}.TotalNumberOfBatches),';',Dataset.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_',options{3}.SummationType,'_{',int2str(options{9}.NumberOfTests),'}_[',int2str(options{9}.SmoothKernel),']_',num2str(options{9}.SD_SmoothingFilter),'_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                            end
                                        else
                                            if strcmp(options{9}.Dimension,'3D')
                                                FileName=strcat(savedir,'Train-',int2str(options{1}.TotalNumberOfBatches),';',Dataset.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_',options{3}.SummationType,'_{',int2str(options{9}.NumberOfTests(1)),'}_[NO Smoothing]_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                            elseif strcmp(options{9}.Dimension,'2D') 
                                                FileName=strcat(savedir,'Train-',int2str(options{1}.TotalNumberOfBatches),';',Dataset.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_',options{3}.SummationType,'_{',int2str(options{9}.NumberOfTests),'}_[NO Smoothing]_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                            end
                                        end
                        %autoload function. If file does not exist, decrement the stage (Features to Raw, Expanded Feature to Features etc) and try loading again.
                            if exist(FileName, 'file')
                                    load(FileName)        
                                    if optionsData.Verbose 
                                        disp('Features Loaded Train (No computation)') 
                                    end
                                    Dataset=Features;
                            elseif not(exist(FileName, 'file')) && optionsData.AutoLoad 
                                    options{1}.WhichData='Raw';
                                    [Dataset, options]=LoadDataset(Name,RootPath,options);
                                    disp('Warning: Intended precomputed feature file does not exist')
                            else
                                    msg = 'Loading error occured.';
                                    error(msg)
                            end
                    else %optionsData.TrainTest='Test';
                            if options{9}.SmoothingFlag
                                if strcmp(options{9}.Dimension,'3D')
                                    FileName=strcat(savedir,'Test-',int2str(options{1}.TotalNumberOfBatches),';',Dataset.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_',options{3}.SummationType,'_{',int2str(options{9}.NumberOfTests(1)),'}_[',int2str(options{9}.SmoothKernel),']_',num2str(options{9}.SD_SmoothingFilter),'_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                elseif strcmp(options{9}.Dimension,'2D')
                                    FileName=strcat(savedir,'Test-',int2str(options{1}.TotalNumberOfBatches),';',Dataset.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_',options{3}.SummationType,'_{',int2str(options{9}.NumberOfTests),'}_[',int2str(options{9}.SmoothKernel),']_',num2str(options{9}.SD_SmoothingFilter),'_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                end
                            else
                                if strcmp(options{9}.Dimension,'3D')
                                    FileName=strcat(savedir,'Test-',int2str(options{1}.TotalNumberOfBatches),';',Dataset.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_',options{3}.SummationType,'_{',int2str(options{9}.NumberOfTests(1)),'}_[NO Smoothing]_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                elseif strcmp(options{9}.Dimension,'2D') 
                                    FileName=strcat(savedir,'Test-',int2str(options{1}.TotalNumberOfBatches),';',Dataset.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_',options{3}.SummationType,'_{',int2str(options{9}.NumberOfTests),'}_[NO Smoothing]_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
                                end
                            end

                            load(FileName)        
                            if optionsData.Verbose 
                                disp('Features Loaded Test (No computation)') 
                            end
                            Dataset=Features;
                    end
            end
        
    elseif strcmp(options{2}.Type,'ExtractPoolNormalizeSingleLayerFeature_v3')
            if strcmp(optionsData.TrainTest,'Train')
                FilePrefix=strcat('FeaturesPooledNormalizedTrain_',int2str(optionsData.BatchNumber),'_');
                FileName=strcat(fullfile(optionsNormalize.SavePath,strcat(FilePrefix,optionsData.DatasetName,'_NoRFs',int2str(options{8}.NumberOfRFs),'_RFSize',int2str(options{8}.rfSize),'_Sp',int2str(int8(optionsExtract.SparsifyFlag)),...
                    '_SMp',int2str(100*optionsExtract.SparsityMultiplier),'_',optionsPool.SummationType,'_Bin',int2str(int8(optionsNormalize.Binarize)))),'.mat');
                %autoload function. If file does not exist, decrement the stage (Features to Raw, Expanded Feature to Features etc) and try loading again.
                    if exist(FileName, 'file')
                        load(FileName);
                        if optionsData.Verbose 
                           disp('Features Loaded Test (No computation)') 
                        end
                    Dataset=NeuralNetFeatures;
                    elseif not(exist(FileName, 'file')) && optionsData.AutoLoad 
                            options{1}.WhichData='Raw';
                            [Dataset, options]=LoadDataset(Name,RootPath,options);
                            disp('Warning: Intended precomputed feature file does not exist')
                    else
                            msg = 'Loading error occured.';
                            error(msg)
                    end
            else    % optionsData.TrainTest='Test'
                    FilePrefix=strcat('FeaturesPooledNormalizedTest_',int2str(optionsData.BatchNumber),'_'); 
                    %no need for autoload check. If the train file is created then,
                    %test file also exist (unless deleted).
                    FileName=fullfile(optionsNormalize.SavePath,strcat(FilePrefix,optionsData.DatasetName,'_NoRFs',int2str(options{8}.NumberOfRFs),'_RFSize',int2str(options{8}.rfSize),'_Sp',int2str(int8(optionsExtract.SparsifyFlag)),'_SMp',int2str(100*optionsExtract.SparsityMultiplier),'_',optionsPool.SummationType,'_Bin',int2str(int8(optionsNormalize.Binarize))));        
                    load(FileName)        
                            if optionsData.Verbose 
                                disp('Features Loaded Test (No computation)') 
                            end
                    Dataset=NeuralNetFeatures;
            end
    end
elseif strcmp(optionsData.WhichData,'ExpandedFeatures') %expanded features
    %in order to load expanded features, the options should be set
    %accordingly, i.e. optionsExpansion.Perform should be true
    if not(optionsExpansion.Perform)
        msg = 'You want to load Expanded Features, but options does not allow that...';
        error(msg)
    end
 %load the corresponding data depending on the choice:
    %optionsData.TrainTest, i.e. train or test data
    if strcmp(optionsData.TrainTest,'Train')
       FilePrefix=strcat('ExpandedFeaturesTrain_',int2str(optionsData.BatchNumber),'_');
       FileName=strcat(fullfile(optionsExpansion.SavePath,strcat(FilePrefix,optionsData.DatasetName,'_NoRFs',int2str(options{8}.NumberOfRFs),'_RFSize',int2str(options{8}.rfSize),'_Sp',int2str(int8(optionsExtract.SparsifyFlag)),...
        '_SMp',int2str(100*optionsExtract.SparsityMultiplier),'_',optionsPool.SummationType,'_Bin',int2str(int8(optionsNormalize.Binarize)),'E',int2str(int8(optionsExpansion.Perform)),'ET',optionsExpansion.Type)),'.mat');
        %autoload function. If file does not exist, decrement the stage (Features to Raw, Expanded Feature to Features etc) and try loading again.
        if exist(FileName, 'file')
            load(FileName)    
            if optionsData.Verbose 
                disp('Expanded Features Loaded Train (No computation)') 
            end
            Dataset=ExpandedFeatures;
        elseif not(exist(FileName, 'file')) && optionsData.AutoLoad 
            options{1}.WhichData='Features';
            [Dataset,options]=LoadDataset(Name,RootPath,options);
            disp('Warning: Intended precomputed expanded feature file does not exist')
        else
            msg = 'Loading error occured.';
            error(msg)
        end     
    else
        %no need for autoload check. If the train file is created then,
        %test file also exist (unless deleted).
        FilePrefix=strcat('ExpandedFeaturesTest_',int2str(optionsData.BatchNumber),'_');
         load(fullfile(optionsExpansion.SavePath,strcat(FilePrefix,optionsData.DatasetName,'_NoRFs',int2str(options{8}.NumberOfRFs),'_RFSize',int2str(options{8}.rfSize),'_Sp',int2str(int8(optionsExtract.SparsifyFlag)),...
        '_SMp',int2str(100*optionsExtract.SparsityMultiplier),'_',optionsPool.SummationType,'_Bin',int2str(int8(optionsNormalize.Binarize)),'E',int2str(int8(optionsExpansion.Perform)),'ET',optionsExpansion.Type)))               
        if optionsData.Verbose 
            disp('Expanded Features Loaded Train (No computation)') 
        end
        Dataset=ExpandedFeatures;
    end
elseif strcmp(optionsData.WhichData,'Model') %learned model via supervised learning
    if strcmp(optionsData.TrainTest,'Train')
       FilePrefix=strcat('Model_',int2str(optionsData.BatchNumber),'_');
       FileName=strcat(fullfile(optionsSupervisedLearning.SavePath,strcat(FilePrefix,optionsData.DatasetName,'_NoRFs',int2str(options{8}.NumberOfRFs),'_RFSize',int2str(options{8}.rfSize),'_Sp',int2str(int8(optionsExtract.SparsifyFlag)),...
        '_SMp',int2str(100*optionsExtract.SparsityMultiplier),'_',optionsPool.SummationType,'_Bin',int2str(int8(optionsNormalize.Binarize)))),'.mat');
        %autoload function. If file does not exist, decrement the stage (Features to Raw, Expanded Feature to Features etc) and try loading again.
        if exist(FileName, 'file')
            load(FileName)   
            if optionsData.Verbose 
                disp('Model Loaded (No computation)') 
            end
            Dataset=Model;
        elseif not(exist(FileName, 'file')) && optionsData.AutoLoad 
            if optionsExpansion.Perform
                options{1}.WhichData='ExpandedFeatures';
            else
                options{1}.WhichData='Features';
            end
            [Dataset,options]=LoadDataset(Name,RootPath,options);
            disp('Warning: Intended precomputed feature file does not exist')
        else
            msg = 'Loading error occured.';
            error(msg)
        end     
        
    elseif strcmp(optionsData.TrainTest,'Test') && optionsExpansion.Perform %during test, do not load the model, load the expanded features.
        FilePrefix=strcat('ExpandedFeaturesTest_',int2str(optionsData.BatchNumber),'_');
        FileName=strcat(fullfile(optionsExpansion.SavePath,strcat(FilePrefix,optionsData.DatasetName,'_NoRFs',int2str(options{8}.NumberOfRFs),'_RFSize',int2str(options{8}.rfSize),'_Sp',int2str(int8(optionsExtract.SparsifyFlag)),...
        '_SMp',int2str(100*optionsExtract.SparsityMultiplier),'_',optionsPool.SummationType,'_Bin',int2str(int8(optionsNormalize.Binarize)),'E',int2str(int8(optionsExpansion.Perform)),'ET',optionsExpansion.Type)),'.mat');
        %autoload function. If file does not exist, decrement the stage (Features to Raw, Expanded Feature to Features etc) and try loading again.
        if exist(FileName, 'file')
            load(FileName)      
            if optionsData.Verbose 
                sprintf('Expanded Features Loaded Test (No computation)') 
            end
            Dataset=ExpandedFeatures;
        elseif not(exist(FileName, 'file')) && optionsData.AutoLoad 
            options{1}.WhichData='Features';
            [Dataset,options]=LoadDataset(Name,RootPath,options);
            sprintf('Warning: Intended precomputed expanded feature file does not exist')
        else
            msg = 'Loading error occured.';
            error(msg)
        end          
    elseif strcmp(optionsData.TrainTest,'Test') && not(optionsExpansion.Perform) %during test, do not load the model, load the features.
        FilePrefix=strcat('FeaturesPooledNormalizedTest_',int2str(optionsData.BatchNumber),'_');
        FileName=strcat(fullfile(optionsNormalize.SavePath,strcat(FilePrefix,optionsData.DatasetName,'_NoRFs',int2str(options{8}.NumberOfRFs),'_RFSize',int2str(options{8}.rfSize),'_Sp',int2str(int8(optionsExtract.SparsifyFlag)),...
            '_SMp',int2str(100*optionsExtract.SparsityMultiplier),'_',optionsPool.SummationType,'_Bin',int2str(int8(optionsNormalize.Binarize)))),'.mat');
        %autoload function. If file does not exist, decrement the stage (Features to Raw, Expanded Feature to Features etc) and try loading again.
        if exist(FileName, 'file')
            load(FileName)     
            if optionsData.Verbose 
                disp('Features Loaded Test (No computation)') 
            end
            Dataset=NeuralNetFeatures;
        elseif not(exist(FileName, 'file')) && optionsData.AutoLoad 
            options{1}.WhichData='Raw';
            [Dataset, options]=LoadDataset(Name,RootPath,options);
            disp('Warning: Intended precomputed feature file does not exist')
        else
            msg = 'Loading error occured.';
            error(msg)
        end
    end
end

%save important parameters to the dataset
Dataset.ImageDim=ImageDim;
Dataset.DatasetName=Name;
Dataset.Name=Name;
Dataset.NumberOfBatches=NumberOfBatches;
Dataset.trainY=trainY;
Dataset.testY=testY;
Dataset.UniqueLabels=UniqueLabels;

end %end function

