%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function loads the PRECOMPUTED FEATURES of wide variety of datasets,
%   generally for computer vision/machine learning tasks.
%   CAUTION: can only be used within MemoryEff experiments.
%
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


function Dataset=LoadDatasetFeatures(Name,RootPath,options)

%default options, if not given by the user
if nargin < 3
    options.Randomize=false;
    options.TrainDataRatio=1;
    options.BatchNumber=0;
end

%load dataset paths, saved in a global mat file
load(fullfile(RootPath,'Dataset\DatasetPaths'));

%load raw datasets from the corresponding mat files for each dataset
switch Name
    case 'CIFAR10'
        load(CIFAR10_DataPath);
        ImageDim=[32 32 3];
        
    case 'CIFAR100'
        load(CIFAR100_DataPath);  
        ImageDim=[32 32 3];
        
    case 'STL10'
    
    case 'SUN_Scene'
        
    case 'SUN_SceneAttributes'    
     
    case 'SUN_Objects'
        
    case 'MSCOCO_Objects'
        
    case 'Texture'
    
    case 'Material'
        
    
end

%randomize data if flag is raised in options
if options.Randomize
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
if options.TrainDataRatio<1
    trainY=trainY(1:round(options.TrainDataRatio*size(trainX,1)));
    trainX=trainX(1:round(options.TrainDataRatio*size(trainX,1)),:);
    
    %!!!!!for debugging purposes, remove LATER!!!!
    testY=testY(1:round(options.TrainDataRatio*size(testY,1)));
    testX=testX(1:round(options.TrainDataRatio*size(testX,1)),:);

end

Dataset.options=options;
Dataset.trainX=trainX;
Dataset.trainY=trainY;
Dataset.testX=testX;
Dataset.testY=testY;
Dataset.ImageDim=ImageDim;
Dataset.Name=Name;

%show the progress on command window
fprintf('Dataset Loaded \n');

end %end function