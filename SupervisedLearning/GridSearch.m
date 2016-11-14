%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function performs grid search of optimum parameter for linear
%supervised learning.
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


function [lambda_optimum, rbfScale_optimum]=GridSearch(InitW,Features,Labels,nClasses,options,options_minFunc)

%extract important paramaters of the experiment
nVars=size(Features,2)-1;

%initialize, so that return with originals if not Softmax or KernelSVM methods
lambda_optimum=options.lambda;
rbfScale_optimum=options.rbfScale;

%narrow down the dataset size for fast grid search
UsedSize=round(options.GridSearchDataRatio*size(Features,1));
Features=Features(1:UsedSize,:);
Labels=Labels(1:UsedSize);
%divide the data to train and test. 80-20 split, hardwired.
TrainDataSize=round(0.8*size(Features,1));

if strcmp(options.ModelType,'Softmax') %only lambda search for Softmax
    %form the objective function for training
    funObjTrain= @(W)SoftmaxLoss2_oy(W,Features(1:TrainDataSize,:),Labels(1:TrainDataSize),nClasses,options.GPU);

    %Coarse search, with large steps of lambda
    SearchIter=1;
    CoarseMult=4;
    for i=-options.GridSearchDepth:1:options.GridSearchDepth
        lambda_iter=(CoarseMult^i)*options.lambda*ones(nVars+1,nClasses-1);
        lambda_iter(1,:) = 0; % Don't penalize biases
        wSoftmax = minFunc_oy(@penalizedL2,InitW,options_minFunc,funObjTrain,lambda_iter(:));
        wSoftmax = reshape(wSoftmax,[nVars+1 nClasses-1]);
        wSoftmax = [wSoftmax zeros(nVars+1,1)];
        [~, yhat] = max(Features(TrainDataSize+1:end,:)*wSoftmax,[],2);
        testAccuracy(SearchIter,:) = [(CoarseMult^i)*options.lambda 1-sum(yhat~=Labels(TrainDataSize+1:end))/length(Labels(TrainDataSize+1:end))];
        SearchIter=SearchIter+1;
    end
    [MaxVal, MaxInd]=max(testAccuracy(:,2));
    lambda_optimum=gather(testAccuracy(MaxInd,1));

    %if fine search is ON, then do another round with smaller steps
    %this time around lambda_optimum, found above.
    FineMult=1.32;
    if options.GridSearch_Fine
        SearchIter=1;
        for i=-options.GridSearchDepth:1:options.GridSearchDepth
            lambda_iter=(FineMult^i)*lambda_optimum*ones(nVars+1,nClasses-1);
            lambda_iter(1,:) = 0; % Don't penalize biases
            wSoftmax = minFunc_oy(@penalizedL2,InitW,options_minFunc,funObjTrain,lambda_iter(:));
            wSoftmax = reshape(wSoftmax,[nVars+1 nClasses-1]);
            wSoftmax = [wSoftmax zeros(nVars+1,1)];
            [~, yhat] = max(Features(TrainDataSize+1:end,:)*wSoftmax,[],2);
            testAccuracyFine(SearchIter,:) = [(FineMult^i)*lambda_optimum 1-sum(yhat~=Labels(TrainDataSize+1:end))/length(Labels(TrainDataSize+1:end))];
            SearchIter=SearchIter+1;
        end
        [MaxVal, MaxInd]=max(testAccuracyFine(:,2));
        lambda_optimum=gather(testAccuracyFine(MaxInd,1));
    end

elseif strcmp(options.ModelType,'KernelSVM') %lambda and rbfScale search for Kernel
    Krbf = zeros(TrainDataSize,TrainDataSize);
    %form the objective function for training
    funObjTrain = @(W)SoftmaxLoss2_oy(W,Krbf,Labels(1:TrainDataSize),nClasses,options.GPU);
    InitW=randn(TrainDataSize*(nClasses-1),1);
    %Coarse search, with large steps of lambda and rbfScale
    SearchIter=1;
    CoarseMult=5;
    for i=-options.GridSearchDepth:1:options.GridSearchDepth
        for j=-options.GridSearchDepth:1:options.GridSearchDepth
            lambda_iter=(CoarseMult^i)*options.lambda;
            rbfScale_iter=(CoarseMult^j)*options.rbfScale;
            Krbf = kernelRBF(Features(1:TrainDataSize,:),Features(1:TrainDataSize,:),rbfScale_iter);
            funObjTrain = @(W)SoftmaxLoss2_oy(W,Krbf,Labels(1:TrainDataSize),nClasses,options.GPU);
            
            uRBF = minFunc_oy(@penalizedKernelL2_matrix_oy,InitW,options_minFunc,Krbf,nClasses-1,funObjTrain,lambda_iter,options.GPU);
            uRBF = reshape(uRBF,[TrainDataSize nClasses-1]);
            uRBF = [uRBF zeros(TrainDataSize,1)];      
            
            %find support vectors that satisfy a certain weight
            MaxWeights=max(abs(uRBF),[],2);
            %sort weights
            [~, MW_Index]=sort(MaxWeights,'descend');
            %and choose the best options.SV_Ratio of them
            SVIndex=MW_Index(1:options.SV_Ratio*length(MW_Index));
            SupportVectors=Features(SVIndex,:);
            %modify linear model accordingly
            uRBF=uRBF(SVIndex,:);
            
            KrbfTest = kernelRBF(Features(TrainDataSize+1:end,:),SupportVectors,rbfScale_iter);
                
            [~, yhat] = max(KrbfTest*uRBF,[],2);
            testAccuracy(SearchIter,:) = [(CoarseMult^i)*options.lambda (CoarseMult^j)*options.rbfScale 1-sum(yhat~=Labels(TrainDataSize+1:end))/length(Labels(TrainDataSize+1:end))];
            SearchIter=SearchIter+1;
        end
    end
    [MaxVal, MaxInd]=max(testAccuracy(:,3));
    lambda_optimum=gather(testAccuracy(MaxInd,1));
    rbfScale_optimum=gather(testAccuracy(MaxInd,2));
    
    %if fine search is ON, then do another round with smaller steps
    %this time around lambda_optimum, found above.
    FineMult=2;
    if options.GridSearch_Fine
        %Coarse search, with large steps of lambda and rbfScale
        SearchIter=1;
        for i=-options.GridSearchDepth:1:options.GridSearchDepth
            for j=-options.GridSearchDepth:1:options.GridSearchDepth
                lambda_iter=(FineMult^i)*lambda_optimum;
                rbfScale_iter=(FineMult^j)*rbfScale_optimum;
                Krbf = kernelRBF(Features(1:TrainDataSize,:),Features(1:TrainDataSize,:),rbfScale_iter);

                uRBF = minFunc_oy(@penalizedKernelL2_matrix_oy,InitW,options_minFunc,Krbf,nClasses-1,funObjTrain,lambda_iter,options.GPU);
                uRBF = reshape(uRBF,[TrainDataSize nClasses-1]);
                uRBF = [uRBF zeros(TrainDataSize,1)];      

                %find support vectors that satisfy a certain weight
                MaxWeights=max(abs(uRBF),[],2);
                %sort weights
                [~, MW_Index]=sort(MaxWeights,'descend');
                %and choose the best options.SV_Ratio of them
                SVIndex=MW_Index(1:options.SV_Ratio*length(MW_Index));
                SupportVectors=Features(SVIndex,:);
                %modify linear model accordingly
                uRBF=uRBF(SVIndex,:);

                KrbfTest = kernelRBF(Features(TrainDataSize+1:end,:),SupportVectors,rbfScale_iter);

                [~, yhat] = max(KrbfTest*uRBF,[],2);
                testAccuracyFine(SearchIter,:) = [(FineMult^i)*options.lambda (FineMult^j)*options.rbfScale 1-sum(yhat~=Labels(TrainDataSize+1:end))/length(Labels(TrainDataSize+1:end))];
                SearchIter=SearchIter+1;
            end
        end
        [MaxVal, MaxInd]=max(testAccuracyFine(:,3));
        lambda_optimum=gather(testAccuracyFine(MaxInd,1));
        rbfScale_optimum=gather(testAccuracyFine(MaxInd,2));
    end
    
end

