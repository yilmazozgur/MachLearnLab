%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function evaluates the performance of a model (classification, regression)
%for vision/machine learning tasks.
%
%%% Input:
%Model : A struct of the model created by SupervisedLearning.m
%NormalizedFeatures: A struct created by NormalizeBinarize.m that holds the
%   features of test dataset.
%Labels: labels of the test dataset for evaluation.
%options: struct for options. 
%options.Task= "Classification or Regression?" flag. Default='Classification';
%options.ModelType: The error model to be used in training. Default='Softmax';
%options.GPU: enable GPU computing. Default=false.
%options.SaveFeatures: Save the results to the Root when finished.
%   Default=true.
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
%Ref: It is based on minFunc documantation
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function Performance=EvaluateSupervisedLearning(Model_Batches,NormalizedFeatures,options)

%load options of every stage in a separate struct
optionsData=options{1};
optionsExtract=options{2};
optionsPool=options{3};
optionsNormalize=options{4};
optionsExpansion=options{5};
optionsSupervisedLearning=options{6};
optionsEvaluation=options{7};

%measure computation time
tic

%default options, if not given by the user
if nargin < 2
    optionsEvaluation.Task='Classification';
    optionsEvaluation.ModelType='Softmax';
    optionsEvaluation.GPU=false;
    optionsEvaluation.SavePerformance=true;
    optionsEvaluation.SavePath='F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\SavedData';
end
%extract the some parameters on experiment for bookkeeping
ReceptiveFields=NormalizedFeatures.RFData;
%infer knowledge on the RFs
numCentroids = size(ReceptiveFields.RFs,1);
rfSize=ReceptiveFields.RFSize;
Labels=NormalizedFeatures.Labels;

nClasses = length(NormalizedFeatures.UniqueLabels); %length(unique(Labels));
nVars=size(NormalizedFeatures.Features,2);
nInstances=size(NormalizedFeatures.Features,1);

%default model is the last one in the batch (that sees the whole data)
Model=Model_Batches{end};

%if task is classification, use proper losses and routines
if strcmp(optionsEvaluation.Task,'Classification')
    if strcmp(optionsEvaluation.ModelType,'Softmax')
        % Add bias
        NormalizedFeatures.Features = [ones(nInstances,1) NormalizedFeatures.Features];
    elseif strcmp(optionsEvaluation.ModelType,'SVM2')
        % Add bias
        NormalizedFeatures.Features = [NormalizedFeatures.Features ones(nInstances,1)];
    end
    
    if strcmp(optionsEvaluation.ModelType,'KernelSVM')
        Krbf = kernelRBF(NormalizedFeatures.Features,Model.SupportVectors,Model.rbfScale_optimum);
        %compute training error
        DecisionMatrix=Krbf*Model.Linear;
        [~, yhat(:,1)] = max(DecisionMatrix,[],2);
        DecisionMatrix(sub2ind(size(DecisionMatrix), 1:length(yhat), yhat'))=-100000;
        [~, yhat(:,2)] = max(DecisionMatrix,[],2);

        testAccuracy(1) = 100*(1-sum(yhat(:,1)~=Labels)/length(Labels));
        testAccuracy(2) = 100*(1-sum(yhat(:,1)~=Labels & yhat(:,2)~=Labels)/length(Labels));
        
        %compute bits per decision
        BPC_Mean = ComputeBPC( DecisionMatrix, Labels );
        
        if optionsData.Verbose 
            fprintf('Test Accuracy one decision: %d \n',testAccuracy(1));
            fprintf('Test Accuracy two decisions: %d \n',testAccuracy(2));
            fprintf('Bits per decision: %d \n',BPC_Mean);
        end
        
    
               
    else
        
        %make ensemble decision, if requested
        if optionsEvaluation.Ensemble
            [yhat, DecisionMatrix]=EnsembleDecision(NormalizedFeatures.Features,Model_Batches,nClasses,options);
            BPC_Mean = ComputeBPC( DecisionMatrix, Labels );
        else
            if optionsSupervisedLearning.FeatureSubsetRatio<1 %if subset selected, perform mask here
                NormalizedFeatures.Features=NormalizedFeatures.Features(:,Model.FeatureSubset);
            end
            DecisionMatrix=NormalizedFeatures.Features*Model.Linear;
            [~, yhat(:,1)] = max(DecisionMatrix,[],2);
            DecisionMatrix(sub2ind(size(DecisionMatrix), 1:length(yhat), yhat'))=-100000;
            [~, yhat(:,2)] = max(DecisionMatrix,[],2);
            %compute bits per decision
            BPC_Mean = ComputeBPC( DecisionMatrix, Labels );
        end
                
        
        %compute test error
        testAccuracy(1) = 100*(1-sum(yhat(:,1)~=Labels)/length(Labels));
        testAccuracy(2) = 100*(1-sum(yhat(:,1)~=Labels & yhat(:,2)~=Labels)/length(Labels));
        if optionsData.Verbose 
            fprintf('Test Accuracy one decision: %d \n',testAccuracy(1));
            fprintf('Test Accuracy two decisions: %d \n',testAccuracy(2));
            fprintf('Bits per decision: %d \n',BPC_Mean);
        end


    end

else %regression
    %coming soon...

end

%also include the RF data, for future reference
Performance.options=options;
Performance.optionsPrev=NormalizedFeatures.optionsPrev;
Performance.Model=Model_Batches;
Performance.TrainAccuracy=Model.TrainAccuracy;
Performance.TestAccuracy=testAccuracy;
Performance.BPC_Mean=BPC_Mean;
Performance.RFData=NormalizedFeatures.RFData;
Performance.nClasses=nClasses;
Performance.nFeatures=nVars;
%bookkeeping of the choices
Performance.Mean=NormalizedFeatures.Mean;
Performance.Std=NormalizedFeatures.Std;

%save if instructed
if optionsEvaluation.SavePerformance
    save(fullfile(optionsEvaluation.SavePath,strcat('Performance_',NormalizedFeatures.RFData.DatasetName,'_NoRFs',int2str(numCentroids),'_RFSize',int2str(rfSize),'_Sp',int2str(int8(NormalizedFeatures.SparsifyFlag)),...
        '_SMp',int2str(100*NormalizedFeatures.SparsityMultiplier),'_',NormalizedFeatures.SummationType,'_Bin',int2str(int8(NormalizedFeatures.Binarize)))),'Performance')
    if optionsData.Verbose 
        fprintf('Performance Evaluation saved. \n'); 
    end
end

%show the finalization on command window
if optionsData.Verbose 
    fprintf('Performance Evaluation done!:   %d \n',toc); 
end

end%end function


