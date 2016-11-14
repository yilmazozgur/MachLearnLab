function [ BPC_Mean ] = ComputeBPC( DecisionMatrix, Labels )
%UNTÝTLED Summary of this function goes here
%   Detailed explanation goes here

    %calculate bits per character from Classifier output
    %matrix
    %make the lowest value equal to zero
    
%     %A general method for normalization
%     PredictMat=bsxfun(@minus,DecisionMatrix,min(DecisionMatrix,[],2)); 
    
    %logistic method for normalization
    PredictMat=exp(DecisionMatrix);
    
    PredictMat_Sum=sum(PredictMat,2);
    %compute a probability assignment from the SVM output by
    %dividing to the sum
    ProbMat=bsxfun(@rdivide,PredictMat,PredictMat_Sum);
    %compute the probability of the correct class 
    ProbCorrect=zeros(1,size(Labels,1));
    for ii=1:1:size(Labels,1)
        ProbCorrect(ii)=ProbMat(ii,Labels(ii));
    end
    %the definition of BPC (regularized with a very small value to avoid NaN)
    BitsPerCharAll=-log2(ProbCorrect+0.0001);
    BPC_Mean=mean(BitsPerCharAll);
end

