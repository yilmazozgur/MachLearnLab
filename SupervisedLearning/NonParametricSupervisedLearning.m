function NonParametricSupervisedLearning(RootPath,DatasetName,ReceptiveFields,optionsAll)
optionsSupervisedLearning=optionsAll{6};
k=optionsAll{11}.k;
build_params.algorithm=optionsAll{11}.build_params_algorithm;
build_params.target_precision=optionsAll{11}.build_params_target_precision;
build_params.build_weight=optionsAll{11}.build_params_build_weight;
build_params.build_memory_weight=optionsAll{11}.build_memory_weight;
build_params.build_sample_fraction=optionsAll{11}.build_sample_fraction;


%Loading the entire extracted features and save them into one matrix
for i=1:optionsAll{1,1}.TotalNumberOfBatches%optionsAll{1}.TotalNumberOfBatches
        optionsAll{1}.TrainTest='Train';
        optionsAll{1}.BatchNumber=i;
        tic
        [Dataset, optionsAll]=LoadDataset(DatasetName,RootPath,optionsAll);
        fprintf(strcat(int2str(i),'th Batch is Loaded : '));
        toc
         Train_BRIEF_Features = LabExtractFeatures(Dataset,ReceptiveFields,[],optionsAll);
        DataStats.Mean=Train_BRIEF_Features.Mean;
        DataStats.Std=Train_BRIEF_Features.Std;

         
         optionsAll{1}.TrainTest='Test';
         [Dataset, optionsAll]=LoadDataset(DatasetName,RootPath,optionsAll);
         Test_BRIEF_Features = LabExtractFeatures(Dataset,ReceptiveFields,[],optionsAll);
         batch_length_test=size(Test_BRIEF_Features.Features,1);
         TestSet((i-1)*batch_length_test+1:i*batch_length_test,:)=Test_BRIEF_Features.Features;
         Test_GroundTruth_Labels((i-1)*batch_length_test+1:i*batch_length_test,:)=Test_BRIEF_Features.Labels;
         
         %Splitting the Train Features into train and cross-validation parts
         nInstances=size(Train_BRIEF_Features.Features,1);
         batch_length_train=size(Train_BRIEF_Features.Features,1);
         if optionsSupervisedLearning.EpochCrossValidation
                nInstances=round(nInstances*optionsSupervisedLearning.EpochTrainSplitRatio); %change the meaning of nInstances as the number of training data points available after validation split.
                CrossValidationset((i-1)*(batch_length_train-nInstances)+1:i*(batch_length_train-nInstances),:)=Train_BRIEF_Features.Features(nInstances+1:end,:);
                CV_GroundTruth_Labels((i-1)*(batch_length_train-nInstances)+1:i*(batch_length_train-nInstances),:)=Train_BRIEF_Features.Labels(nInstances+1:end);
                TrainSet((i-1)*nInstances+1:i*nInstances,:)=Train_BRIEF_Features.Features(1:nInstances,:);
                Train_GroundTruth_Labels((i-1)*nInstances+1:i*nInstances,:)=Train_BRIEF_Features.Labels(1:nInstances);
         end   
end

%Search for k-Nearest Neighbors for Cross-Validation features 

TrainsetBatcheNumber=optionsAll{11}.TrainsetBatcheNumber;
SplitTrain=45000/TrainsetBatcheNumber;

for j=1:TrainsetBatcheNumber
    tic
    [index,parameters,speedup]=flann_build_index(TrainSet((j-1)*SplitTrain+1:j*SplitTrain,:)', build_params);
    [result, dists] = flann_search(TrainSet((j-1)*SplitTrain+1:j*SplitTrain,:)', CrossValidationset', k, parameters);
    predictedLabels=Train_GroundTruth_Labels(result')';
        if k==1
            decidedLabels(j,:)=predictedLabels;
        else
            decidedLabels((j-1)*k+1:j*k,:)=predictedLabels;
        end
        fprintf(strcat(int2str(j),'th Batch is predicted : '));
    toc
end
decidedLabels=mode(decidedLabels);
B=[];
B=find(decidedLabels'-CV_GroundTruth_Labels==0);
K_value=length(B);
 fprintf('___________________________________________Cross-Validation Accuracy for k = %d is : %f',k,100*length(B)/length(CV_GroundTruth_Labels));


%  for j=1:50
%     tic
%     [index,parameters,speedup]=flann_build_index(TrainSet((j-1)*SplitTrain+1:j*SplitTrain,:)', build_params);
%     [result, dists] = flann_search(TrainSet((j-1)*SplitTrain+1:j*SplitTrain,:)', TestSet', k, parameters);
%     predictedLabels=Test_GroundTruth_Labels(result')';
%         if k==1
%         decidedLabels(j,:)=predictedLabels;
%         else
%         decidedLabels((j-1)*k+1:j*k,:)=predictedLabels;
%         end
%         fprintf(strcat(int2str(j),'th Batch is predicted : '));
%     toc
%  end
% decidedLabels=mode(decidedLabels);
% B=[];
% B=find(decidedLabels'-Test_GroundTruth_Labels==0);
% K_value(k)=length(B);
%  fprintf('___________________________________________Test Accuracy for k = %d is : %f',k,100*length(B)/length(Test_GroundTruth_Labels));


%

end