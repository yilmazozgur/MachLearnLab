%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function makes an ensemble classification decision given multiple
%models. See the mother function, EvaluateSupervisedLearning.m, for
%details.
%
%
%
%From:
%TOU_ML
%Ozgur Yilmaz, Turgut Ozal University, Ankara
%Web: ozguryilmazresearch.net
%Ref: It is based on minFunc documantation
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [yhat, DecisionMatrix]=EnsembleDecision(Features,Model_Batches,nClasses,options)
nInstances=size(Features,1);

Features(:,1)=[]; %lack of this line caused a very hard to find bug!!!!!


%find the decision of each model in a loop
for i=1:1:options{1,1}.NumberOfBatches
    Model=Model_Batches{i};
    %if subset selection is performed during training, do it here also with
    %right mask
    
    if options{1,6}.FeatureSubsetRatio<1 
        Features_iter=Features(:,Model.FeatureSubset);
    else
        Features_iter=Features;
    end
    Features_iter = [ones(nInstances,1) Features_iter];
    
    %create the decision matrix, for BPC computation
    if i==1
        DecisionMatrix=Features_iter*Model.Linear;
    else
        DecisionMatrix=DecisionMatrix+Features_iter*Model.Linear;
    end
    
    [~, yhat] = max(Features_iter*Model.Linear,[],2);
    YhatMat(i,:)=yhat;
end


if strcmp(options{1,7}.EnsembleDecisionType,'Mean')
    yhat=(round(mean(YhatMat)))';
    yhat=[yhat yhat];
elseif strcmp(options{1,7}.EnsembleDecisionType,'Median')
    yhat=(median(YhatMat))';
    yhat=[yhat yhat];
else %majority voting
   for ii=1:1:size(YhatMat,2) %apply majority voting
        LabelOneInput=YhatMat(:,ii);
        [HistCount, ~]=hist(LabelOneInput,1:nClasses);
        [~, MaxIndex]=max(HistCount);
        yhat(ii,1)=MaxIndex;
        HistCount(MaxIndex)=0;
        [~, MaxIndex]=max(HistCount);
        yhat(ii,2)=MaxIndex;
   end
%    yhat=yhat';
end



end %function end