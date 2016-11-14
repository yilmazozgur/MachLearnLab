%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function performs early stopping for linear supervised learning.
%See the mother function, SupervisedLearning.m for Input/Output details,
%and options.
%
%
%From:
%TOU_ML
%Ozgur Yilmaz, Turgut Ozal University, Ankara
%Web: ozguryilmazresearch.net
%Ref: It is based on minFunc documantation
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [WLinear, testAccuracy, SVIndex]=EarlyStopping(InitW,Features,Labels,nClasses,options,options_minFunc)

%extract important paramaters of the experiment
nVars=size(Features,2)-1;

%the max number of iterations for no early stopping case.
%this will be used to stop early stopping for max iter bound.
MaxIterHolistic=options_minFunc.MaxIter;

%reduce the number of optimization steps at every iteration of early
%stopping
options_minFunc.MaxIter=options.EarlyStoppingMaxIter;
options_minFunc.MaxFunEvals=options.EarlyStoppingMaxFunEvals;


%divide the data to train and test. 80-20 split, hardwired.
TrainDataSize=round(0.8*size(Features,1));

if strcmp(options.ModelType,'Softmax')
    %form the objective function for training
    funObjTrain= @(W)SoftmaxLoss2_oy(W,Features(1:TrainDataSize,:),Labels(1:TrainDataSize),nClasses,options.GPU);
    lambda=options.lambda*ones(nVars+1,nClasses-1);
    lambda(1,:) = 0; % Don't penalize biases
    %do an initial training
    WLinear = minFunc_oy(@penalizedL2,InitW,options_minFunc,funObjTrain,lambda(:));
    WLinear = reshape(WLinear,[nVars+1 nClasses-1]);
    WLinear = [WLinear zeros(nVars+1,1)];
    [~, yhat] = max(Features(TrainDataSize+1:end,:)*WLinear,[],2);
    testAccuracy = 100*(1-sum(yhat~=Labels(TrainDataSize+1:end))/length(Labels(TrainDataSize+1:end)));
    testAccuracy_Iter=testAccuracy;
else
    InitW=randn(TrainDataSize*(nClasses-1),1);
    %training objective function
    funObjTrain = @(W)SoftmaxLoss2_oy(W,Features(1:TrainDataSize,1:TrainDataSize),Labels(1:TrainDataSize),nClasses,options.GPU);
    lambda=options.lambda;
    %do an initial training
    WLinear = minFunc_oy(@penalizedKernelL2_matrix_oy,InitW,options_minFunc,Features(1:TrainDataSize,1:TrainDataSize),nClasses-1,funObjTrain,lambda,options.GPU);
    WLinear = reshape(WLinear,[TrainDataSize nClasses-1]);
    WLinear = [WLinear zeros(TrainDataSize,1)];
    
    Features_Test=Features(TrainDataSize+1:end,1:TrainDataSize);
    [~, yhat] = max(Features_Test*WLinear,[],2);
    testAccuracy =  100*(1-sum(yhat~=Labels(TrainDataSize+1:end))/length(Labels(TrainDataSize+1:end)));
    testAccuracy_Iter=testAccuracy;
end


SignificantImprovement=true;
TotalImprovementFailure=0;
IterTotal=options.EarlyStoppingMaxIter;
WLinear_Prev=WLinear;
testAccuracy_Iter_Prev=testAccuracy_Iter;
%stop if not enough improvement, or max iter bound is passed
while SignificantImprovement && IterTotal<MaxIterHolistic
    InitW=WLinear;
    InitW(:,end)=[];
    InitW=InitW(:);
    
    if strcmp(options.ModelType,'Softmax')
        WLinear = minFunc_oy(@penalizedL2,InitW,options_minFunc,funObjTrain,lambda(:));
        WLinear = reshape(WLinear,[nVars+1 nClasses-1]);
        WLinear = [WLinear zeros(nVars+1,1)];
        [~, yhat] = max(Features(TrainDataSize+1:end,:)*WLinear,[],2);
        %computer test error
        testAccuracy_Iter = 100*(1-sum(yhat~=Labels(TrainDataSize+1:end))/length(Labels(TrainDataSize+1:end)));
    else
        WLinear = minFunc_oy(@penalizedKernelL2_matrix_oy,InitW,options_minFunc,Features(1:TrainDataSize,1:TrainDataSize),nClasses-1,funObjTrain,lambda,options.GPU);
        WLinear = reshape(WLinear,[TrainDataSize nClasses-1]);
        WLinear = [WLinear zeros(TrainDataSize,1)];

        Features_Test=Features(TrainDataSize+1:end,1:TrainDataSize);
        [~, yhat] = max(Features_Test*WLinear,[],2);
        testAccuracy_Iter =  100*(1-sum(yhat~=Labels(TrainDataSize+1:end))/length(Labels(TrainDataSize+1:end)));
    end
    %record if there is not enough improvement
    if (testAccuracy_Iter-testAccuracy)< options.EarlyStoppingImpThreshold 
        TotalImprovementFailure=TotalImprovementFailure+1;
    else
        TotalImprovementFailure=0; %reset if no decrement
        testAccuracy=testAccuracy_Iter;
        %record the last solution
        WLinear_Prev=WLinear;
        testAccuracy_Iter_Prev=testAccuracy_Iter;
    end
    
    %if enough decrement seen, exit training
    if TotalImprovementFailure>=options.EarlyStoppingFailureAllowed
        SignificantImprovement=false; 
    end
    
    %return the 2nd last solution, in case there was a decrement in
    %performance, i.e. testAccuracy_Iter-testAccuracy<0
    if not(SignificantImprovement)
        WLinear=WLinear_Prev;
        testAccuracy_Iter=testAccuracy_Iter_Prev;
    end
    
    IterTotal=IterTotal+options.EarlyStoppingMaxIter;
end

%get into right shape for consistency
WLinear(:,end)=[];
WLinear=WLinear(:);
