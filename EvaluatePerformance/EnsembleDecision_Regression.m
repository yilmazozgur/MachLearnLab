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



function [yhat, DecisionMatrix]=EnsembleDecision_Regression(Features,Model_Batches,options)
nInstances=size(Features,1);


%find the decision of each model in a loop
for i=1:1:options{1,1}.NumberOfBatches
    Model=Model_Batches{i};
           
    %create the decision matrix, for BPC computation
    if i==1
        DecisionMatrix=(Model.Linear*Features')';
    else
        DecisionMatrix=DecisionMatrix+(Model.Linear*Features')';
    end

end

yhat=DecisionMatrix/options{1,1}.NumberOfBatches;

end %function end

