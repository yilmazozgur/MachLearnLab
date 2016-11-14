%%% Set global parameters for a single train-test experiment
% memory patterns)
D = 5; % nr of payload symbols
M = 10; % length of memory sequence
T=10;
Ntrain=500;
Ntest=100;
EnlargeInput=0;
BundlePeriod=1;

datasetTrain = genDataMem(T, D, M, Ntrain);
datasetTest = genDataMem(T, D, M, Ntest);

trainX_=datasetTrain(:,1:D+2,:);
trainY_=datasetTrain(:,D+3:end,:);
trainX=[];
trainY=[];
for i=1:1:size(trainX_,3)
    InputTemp=squeeze(trainX_(:,:,i));
    InputTemp=(reshape(InputTemp',[BundlePeriod*size(InputTemp,2),size(InputTemp,1)/BundlePeriod]))';
    ZerosDummy=zeros(size(InputTemp,1),EnlargeInput*size(InputTemp,2));
    InputTemp=[InputTemp ZerosDummy];
    trainX=[trainX; InputTemp];
    Abc=squeeze(trainY_(:,:,i));
%     trainY=[trainY; Abc];
    trainY=[trainY; mod(find(Abc'==1),D+2)];
end

testX_=datasetTest(:,1:D+2,:);
testY_=datasetTest(:,D+3:end,:);
testX=[];
testY=[];
for i=1:1:size(testX_,3)
    InputTemp=squeeze(testX_(:,:,i));
    InputTemp=(reshape(InputTemp',[BundlePeriod*size(InputTemp,2),size(InputTemp,1)/BundlePeriod]))';
    ZerosDummy=zeros(size(InputTemp,1),EnlargeInput*size(InputTemp,2));
    InputTemp=[InputTemp ZerosDummy];
    testX=[testX; InputTemp];
    Abc=squeeze(testY_(:,:,i));
%     testY=[testY; Abc];
    testY=[testY; mod(find(Abc'==1),D+2)];
end
UniqueLabels=[1:D+2]';
i=1;
save('F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\RawData\20BitT10_Batch0.mat','trainX','trainY','testX','testY','UniqueLabels','i')

