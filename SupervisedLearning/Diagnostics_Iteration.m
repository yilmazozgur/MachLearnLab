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


function [testAccuracy, IterationOptimum]=Diagnostics_Iteration(InitW,Features,Labels,nClasses,options,options_minFunc)

%extract important paramaters of the experiment
nVars=size(Features,2)-1;

%the max number of iterations for no early stopping case.
%this will be used to stop early stopping for max iter bound.
MaxIterHolistic=options_minFunc.MaxIter;

%reduce the number of optimization steps at every iteration of early
%stopping
options_minFunc.MaxIter=options.EarlyStoppingMaxIter;
options_minFunc.MaxFunEvals=options.EarlyStoppingMaxFunEvals;


%divide the data to train and test. 90-10 split, hardwired.
TrainDataSize=round(0.9*size(Features,1));

Iter=1;
for ii=options_minFunc.MaxIter:options_minFunc.MaxIter:MaxIterHolistic
    if strcmp(options.ModelType,'Softmax')
        %form the objective function for training
        funObjTrain= @(W)SoftmaxLoss2_oy(W,Features(1:TrainDataSize,:),Labels(1:TrainDataSize),nClasses,options.GPU);
        lambda=options.lambda*ones(nVars+1,nClasses-1);
        lambda(1,:) = 0; % Don't penalize biases
        %do an initial training
        options_minFunc.MaxIter=ii;
        options_minFunc.MaxFunEvals=2*ii;
        WLinear = minFunc_oy(@penalizedL2,InitW,options_minFunc,funObjTrain,lambda(:));
        WLinear = reshape(WLinear,[nVars+1 nClasses-1]);
        WLinear = [WLinear zeros(nVars+1,1)];
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
    end
    [~, yhat] = max(Features(TrainDataSize+1:end,:)*WLinear,[],2);
    testAccuracy(Iter,:) = [ii 100*(1-sum(yhat~=Labels(TrainDataSize+1:end))/length(Labels(TrainDataSize+1:end)))];
    Iter=Iter+1;
end

[MaxVal, MaxInd]=max(testAccuracy(:,2));
IterationOptimum=gather(testAccuracy(MaxInd,1));


