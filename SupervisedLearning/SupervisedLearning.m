%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function performs supervised learning (classification, regression)
%on given features using minFunc toolb�x. 
%
%%% Input:
%Model_Init: a previously trained model for warm start. Empty if none.
%NormalizedFeatures : A struct of features (possibly created by 
%   NormalizeBinarize.m)
%Labels: Labels to be used for training.
%options: struct for options. 
%options.Task= "Classification or Regression?" flag. Default='Classification';
%options.ModelType: The error model to be used in training. Default='Softmax';
%options.lambda: Regularization parameter. Default=1e-4;
%options.rbfScale: scale for rbf kernel. Default=100;
%options.SV_Ratio: Ratio of support vectors in RBF kernel 
%   Default=1;
%options.GridSearch: Flag for grid search in parameters. Default=false
%options.GridSearch_Fine: whether to do fine search during grid search. 
%   Default=true.
%options.GridSearchDepth: The number of iterations for grid search. Default=3;
%options.GridSearchDataRatio: A subset of the data can be used for grid
%   search to speed up. Default=0.25.
%options.EarlyStopping: Flag for early stopping based gradient descent.
%   Default=false.
% options.EarlyStoppingMaxIter: Maximum number of iterations in one cycle of 
%   early stopping. It does this much gradient descent iter, then check. 
%   Default=50;
% options.EarlyStoppingMaxFunEvals: Similar to max iter. Default=100;
% options.EarlyStoppingImpThreshold: If the performance improvement on 
%   validation data is less than this threshold, it stops early. Default=0.1;
%options.MaxIter: maximum minFunc iterations. Default=500;
%options.MaxFunEvals: maximum minFunc function evals. Default=1000;
%options.GPU: enable GPU computing. Default=false.
%options.SaveFeatures: Save the results to the Root when finished.
%   Default=false.
%options.SavePath: the path to save the results if SaveRFs flag is up.
%   Default='';
%
%%% Output:
%Model: A struct that holds the supervised learning model, along with all
%   auxillary data.
%
%
%From:
%TOU_ML
%Ozgur Yilmaz, Turgut Ozal University, Ankara
%Web: ozguryilmazresearch.net
%Ref: It is based on minFunc documantation
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function Model=SupervisedLearning(Model_Init,NormalizedFeatures,options)

%load options of every stage in a separate struct
optionsData=options{1};
optionsExtract=options{2};
optionsPool=options{3};
optionsNormalize=options{4};
optionsExpansion=options{5};
optionsSupervisedLearning=options{6};
optionsEvaluation=options{7};

%if the precomputed expanded features, or model are loaded during LoadDataset,
%return from the function.
if strcmp(optionsData.WhichData,'Model')
    Model=NormalizedFeatures;
    return
end

%measure computation time
tic

%default options, if not given by the user
if nargin < 3
    optionsSupervisedLearning.Task='Classification';
    optionsSupervisedLearning.ModelType='Softmax';
    optionsSupervisedLearning.lambda=1e-4;
    optionsSupervisedLearning.rbfScale=100;
    optionsSupervisedLearning.SV_Ratio=1;
    optionsSupervisedLearning.GridSearch=false;
    optionsSupervisedLearning.GridSearch_Fine=true;
    optionsSupervisedLearning.GridSearchDepth=3;
    optionsSupervisedLearning.GridSearchDataRatio=0.25;
    optionsSupervisedLearning.EarlyStopping=false;
    optionsSupervisedLearning.EarlyStoppingMaxIter=50;
    optionsSupervisedLearning.EarlyStoppingMaxFunEvals=100;
    optionsSupervisedLearning.EarlyStoppingImpThreshold=0.1;
    optionsSupervisedLearning.MaxIter=500;
    optionsSupervisedLearning.MaxFunEvals=1000;
    optionsSupervisedLearning.GPU=false;
    optionsSupervisedLearning.SaveModel=true;
    optionsSupervisedLearning.SavePath='F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\SavedData';
end

%extract the some parameters on experiment for bookkeeping
 ReceptiveFields=NormalizedFeatures.RFData;
%infer knowledge on the RFs
numCentroids = size(ReceptiveFields.RFs,1);
rfSize=ReceptiveFields.RFSize;

%default intitilizations here
SupportVectors=[];
lambda_optimum=optionsSupervisedLearning.lambda;
rbfScale_optimum=optionsSupervisedLearning.rbfScale;
FeatureSubset=[];
Labels=NormalizedFeatures.Labels;

%Show what is to be done
if optionsData.Verbose 
    fprintf('Supervised Learning Start: \n'); 
end

nClasses = length(NormalizedFeatures.UniqueLabels); %length(unique(Labels));
nVars=size(NormalizedFeatures.Features,2);
nInstances=size(NormalizedFeatures.Features,1);

%apply feature subset selection, if ratio is smaller than 1
if optionsSupervisedLearning.FeatureSubsetRatio<1
    PermSubset=randperm(nVars);
    nVars=round(nVars*optionsSupervisedLearning.FeatureSubsetRatio);
    FeatureSubset=PermSubset(1:nVars);
    NormalizedFeatures.Features=NormalizedFeatures.Features(:,FeatureSubset);
end

%if cross validation option is ON, split the train-validation parts 
%it is hard coded as 80-20 split. Does not work for Kernel SVM.
ValidationAccuracy=-1;
Validation_BPC=-1;
if optionsSupervisedLearning.EpochCrossValidation
    nInstances=round(nInstances*optionsSupervisedLearning.EpochTrainSplitRatio); %change the meaning of nInstances as the number of training data points available after validation split.
    ValidationFeatures=NormalizedFeatures.Features(nInstances+1:end,:);
    ValidationLabels=NormalizedFeatures.Labels(nInstances+1:end);
    NormalizedFeatures.Features=NormalizedFeatures.Features(1:nInstances,:);
    NormalizedFeatures.Labels=NormalizedFeatures.Labels(1:nInstances);
    Labels=Labels(1:nInstances);
end

%move data to GPU if flag is ON
if optionsSupervisedLearning.GPU
    NormalizedFeatures.Features=gpuArray(NormalizedFeatures.Features);
    Labels=gpuArray(Labels);
end

%if task is classification, use proper losses and routines
if strcmp(optionsSupervisedLearning.Task,'Classification')
    if strcmp(optionsSupervisedLearning.ModelType,'Softmax')
        % Add bias
        NormalizedFeatures.Features = [ones(nInstances,1) NormalizedFeatures.Features];
        if optionsSupervisedLearning.EpochCrossValidation
            ValidationFeatures = [ones(length(ValidationLabels),1) ValidationFeatures];
        end     
        
        %set objective function
        funObj = @(W)SoftmaxLoss2_oy(W,NormalizedFeatures.Features,Labels,nClasses,optionsSupervisedLearning.GPU);
        lambda = optionsSupervisedLearning.lambda*ones(nVars+1,nClasses-1);
        lambda(1,:) = 0; % Don't penalize biases
        if optionsData.Verbose 
            fprintf('Training multinomial logistic regression model...\n'); 
        end
        
        
        %Harness previously trained model if possible. For warm start.
        if isempty(Model_Init) || optionsSupervisedLearning.FeatureSubsetRatio<1 || optionsEvaluation.Ensemble % if there is feature subset selection or ensemble classification, we can not do warm start!
            InitW=zeros((nVars+1)*(nClasses-1),1);
        else
            InitW=Model_Init.Linear;
            InitW(:,end)=[];
            InitW=InitW(:);
        end
        
        %set minFunc options
        options_minFunc.Display = 0;
        options_minFunc.MaxIter=optionsSupervisedLearning.MaxIter;
        options_minFunc.MaxFunEvals=optionsSupervisedLearning.MaxFunEvals;
%         options_minFunc.Display = 0;
%         options_minFunc.Method='qnewton';
                
        %perform grid search on the regularization parameter, lambda
        if optionsSupervisedLearning.GridSearch
            [lambda_optimum, ~]=GridSearch(InitW,NormalizedFeatures.Features,Labels,nClasses,optionsSupervisedLearning,options_minFunc);
            lambda = lambda_optimum*ones(nVars+1,nClasses-1);   
            lambda(1,:) = 0; % Don't penalize biases
        end
        
        %find the best number of iterations by cross validation
        if optionsSupervisedLearning.IterationDiagnostics
            [testAccuracy, IterationOptimum]=Diagnostics_Iteration(InitW,NormalizedFeatures.Features,Labels,nClasses,optionsSupervisedLearning,options_minFunc);
            options_minFunc.MaxIter=IterationOptimum;
            options_minFunc.MaxFunEvals=2*IterationOptimum;
        end
                
        %perform early stopping if flag is ON, to avoid overfitting
        if optionsSupervisedLearning.EarlyStopping
            optionsES=optionsSupervisedLearning;
            optionsES.lambda=lambda_optimum; %in case grid search result is to be used in early stopping
            [wSoftmax, testAccuracyES]=EarlyStopping(InitW,NormalizedFeatures.Features, Labels, nClasses, optionsES, options_minFunc);
        else
            wSoftmax = minFunc_oy(@penalizedL2,InitW,options_minFunc,funObj,lambda(:));
        end
        
        %reshape it, so that it can directly be used on feature vector.
        wSoftmax = reshape(wSoftmax,[nVars+1 nClasses-1]);
        wSoftmax = [wSoftmax zeros(nVars+1,1)];
        WLinear=wSoftmax; %linear model
        
    elseif strcmp(optionsSupervisedLearning.ModelType,'SVM') %other option is linear SVM. CAUTION: GPU, GridSearch and EarlyStopping is not implemented for SVM.
        % Linear SVM. set objective function
        funObj = @(w)SSVMMultiLoss_oy(w,NormalizedFeatures.Features,Labels,nClasses,optionsSupervisedLearning.GPU);
        lambda = optionsSupervisedLearning.lambda;
        if optionsData.Verbose 
            fprintf('Training linear multi-class SVM...\n'); 
        end
        
        %Harness previously trained model if possible. For warm start.
        if isempty(Model_Init) || optionsSupervisedLearning.FeatureSubsetRatio<1 || optionsEvaluation.Ensemble % if there is feature subset selection, or ensemble classification we can not do warm start!
            InitW=zeros(nVars*nClasses,1);
        else
            InitW=Model_Init.Linear;
            InitW=InitW(:);
        end
        
        %set minFunc options
        options_minFunc.Display = 0;
        options_minFunc.MaxIter=optionsSupervisedLearning.MaxIter;
        options_minFunc.MaxFunEvals=optionsSupervisedLearning.MaxFunEvals;
        wLinear = minFunc_oy(@penalizedL2,InitW,options_minFunc,funObj,lambda);
        wLinear = reshape(wLinear,[nVars nClasses]);
        WLinear=wLinear;%linear model
        
    elseif strcmp(optionsSupervisedLearning.ModelType,'KernelSVM')
        
        if optionsData.Verbose 
            fprintf('Training kernel(rbf) multinomial logistic regression model...\n'); 
        end
        % RBF kernel based softmax
        Krbf = kernelRBF(NormalizedFeatures.Features,NormalizedFeatures.Features,optionsSupervisedLearning.rbfScale);

        if optionsSupervisedLearning.GPU
            Krbf=gpuArray(Krbf);
        end
        
        %Harness previously trained model if possible. For warm start.
        if isempty(Model_Init) || optionsData.NumberOfBatches>0 % if there are more than one batch in the data, they can not help each other for initilazitation.
            InitW=randn(nInstances*(nClasses-1),1);
        else
            InitW=Model_Init.Linear;
            InitW(:,end)=[];
            InitW=InitW(:);
        end
        
        %set minFunc options
        options_minFunc.Display = 0;
        options_minFunc.MaxIter=optionsSupervisedLearning.MaxIter;
        options_minFunc.MaxFunEvals=optionsSupervisedLearning.MaxFunEvals;
        
        %perform grid search on the regularization parameter, lambda and
        %rbfScale
        if optionsSupervisedLearning.GridSearch
            [lambda_optimum, rbfScale_optimum]=GridSearch(InitW,NormalizedFeatures.Features,Labels,nClasses,optionsSupervisedLearning,options_minFunc);
            lambda = lambda_optimum;   
            % RBF kernel based softmax, recompute with
            % right rbfScale, after grid search 
            Krbf = kernelRBF(NormalizedFeatures.Features,NormalizedFeatures.Features,rbfScale_optimum);
        end
        
        %WARNING: does not work when a subset of the SVs are selected via 
        %optionsSupervisedLearning.SV_Ratio<1.
        %perform early stopping if flag is ON, to avoid overfitting
        if optionsSupervisedLearning.EarlyStopping
            optionsES=optionsSupervisedLearning;
            optionsES.lambda=lambda_optimum; %in case grid search result is to be used in early stopping
            optionsES.rbfScale=rbfScale_optimum; %in case grid search result is to be used in early stopping
            [uRBF, testAccuracyES]=EarlyStopping(InitW,Krbf,Labels,nClasses,optionsES,options_minFunc);
            uRBF = reshape(uRBF,[nInstances*0.8 nClasses-1]);
            WLinear = [uRBF zeros(nInstances*0.8,1)];
            
            %choose the best optionsSupervisedLearning.SV_Ratio of vectors, SVIndex comes
            %from early stopping routine
            SupportVectors=NormalizedFeatures.Features(1:nInstances*0.8,:);
            Krbf=Krbf(:,1:nInstances*0.8); 
        else
            funObj = @(W)SoftmaxLoss2_oy(W,Krbf,Labels,nClasses,optionsSupervisedLearning.GPU);
            uRBF = minFunc_oy(@penalizedKernelL2_matrix_oy,InitW,options_minFunc,Krbf,nClasses-1,funObj,lambda_optimum,optionsSupervisedLearning.GPU);
            uRBF = reshape(uRBF,[nInstances nClasses-1]);
            WLinear = [uRBF zeros(nInstances,1)];
            %find support vectors that satisfy a certain weight
            MaxWeights=max(abs(uRBF),[],2);
            %sort weights
            [~, MW_Index]=sort(MaxWeights,'descend');
            %and choose the best optionsSupervisedLearning.SV_Ratio of them
            SVIndex=MW_Index(1:optionsSupervisedLearning.SV_Ratio*length(MW_Index));
            SupportVectors=NormalizedFeatures.Features(SVIndex,:);

            %modify linear model and kernel matrix accordingly
            WLinear=WLinear(SVIndex,:);
            Krbf=Krbf(:,SVIndex);
        end
                
    elseif  strcmp(optionsSupervisedLearning.ModelType,'SVM2')   
     % Add bias
    NormalizedFeatures.Features = [NormalizedFeatures.Features ones(nInstances,1)];
    if optionsSupervisedLearning.EpochCrossValidation
        ValidationFeatures = [ValidationFeatures ones(length(ValidationLabels),1) ];
    end 
     if optionsData.Verbose 
            fprintf('Training multinomial SVM2 model...\n'); 
     end
         %Harness previously trained model if possible. For warm start.
    if isempty(Model_Init) || optionsSupervisedLearning.FeatureSubsetRatio<1 || optionsEvaluation.Ensemble % if there is feature subset selection or ensemble classification, we can not do warm start!
%         InitW=zeros((nVars+1)*(nClasses-1),1);
    
    InitW = zeros((nVars+1)*nClasses,1);
    else
        InitW=Model_Init.Linear;
        InitW(:,end)=[];
        InitW=InitW(:);
    end

    %set minFunc options
    options_minFunc.Display = 0;
    options_minFunc.MaxIter=optionsSupervisedLearning.MaxIter;
    options_minFunc.MaxFunEvals=optionsSupervisedLearning.MaxFunEvals;
        
    C=optionsSupervisedLearning.SVM_C;
    WLinear = train_svm_ourtoolbox(NormalizedFeatures.Features,Labels, C, nClasses,options_minFunc,InitW);
    
    end
    
    
    
        %compute training error
        if strcmp(optionsSupervisedLearning.ModelType,'KernelSVM')
            [~, yhat_train] = max(Krbf*WLinear,[],2);
        else
            [~, yhat_train] = max(NormalizedFeatures.Features*WLinear,[],2);
            if optionsSupervisedLearning.EpochCrossValidation
                [~, yhat_valid] = max(ValidationFeatures*WLinear,[],2);
                ValidationAccuracy=100*(1-sum(yhat_valid~=ValidationLabels)/length(ValidationLabels));
                Validation_BPC = ComputeBPC( ValidationFeatures*WLinear, ValidationLabels);
            end
        end
         TrainAccuracy = gather(100*(1-sum(yhat_train~=Labels)/length(Labels)));
         

    
else %regression
    %coming soon...

end

Model.options=options;
Model.optionsAll=NormalizedFeatures.optionsAll;
Model.Linear=WLinear;
Model.FeatureSubset=FeatureSubset;
Model.SupportVectors=SupportVectors;
Model.lambda_optimum=lambda_optimum;
Model.rbfScale_optimum=rbfScale_optimum;
Model.TrainAccuracy=TrainAccuracy;
Model.ValidationAccuracy=ValidationAccuracy;
Model.Validation_BPC=Validation_BPC;
Model.RFData=NormalizedFeatures.RFData; %also include the RF data, for future reference
Model.Mean=NormalizedFeatures.Mean;
Model.Std=NormalizedFeatures.Std;
Model.ExMean=NormalizedFeatures.ExMean;
Model.ExStd=NormalizedFeatures.ExStd;
Model.RandomPermutation=NormalizedFeatures.RandomPermutation;
Model.SamplingMask=NormalizedFeatures.SamplingMask;

%save if instructed
if optionsSupervisedLearning.SaveModel
    FilePrefix=strcat('Model_',int2str(optionsData.BatchNumber),'_');
    save(fullfile(optionsSupervisedLearning.SavePath,strcat(FilePrefix,NormalizedFeatures.RFData.DatasetName,'_NoRFs',int2str(numCentroids),'_RFSize',int2str(rfSize),'_Sp',int2str(int8(NormalizedFeatures.SparsifyFlag)),...
        '_SMp',int2str(100*NormalizedFeatures.SparsityMultiplier),'_',NormalizedFeatures.SummationType,'_Bin',int2str(int8(NormalizedFeatures.Binarize)))),'Model')
    if optionsData.Verbose 
        fprintf('Learned Model saved. \n'); 
    end
end

%show the finalization on command window
if optionsData.Verbose 
    fprintf('Supervised Learning done!:   %d \n',toc); 
end

end%end function


