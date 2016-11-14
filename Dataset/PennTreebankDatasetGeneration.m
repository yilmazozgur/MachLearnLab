load('F:\Dropbox Folder\Research\Projects\Recurrent Holistic Vision\MatlabCode\Dataset\DatasetPaths.mat')
load('F:\Dropbox Folder\Research\Projects\Classification\LSTMexperimentsMatlab\Language Modeling\Data\TrainData.mat')
load('F:\Dropbox Folder\Research\Projects\Classification\LSTMexperimentsMatlab\Language Modeling\Data\ValidData.mat')
load('F:\Dropbox Folder\Research\Projects\Classification\LSTMexperimentsMatlab\Language Modeling\Data\TestData.mat')
UniqueLabels=UniqueCharacters;


WhichDataType='Binary'; %Binary or Indicator
TrainMatrix=['Train_Ascii_' WhichDataType];
TrainMatrix=eval(TrainMatrix);
ValidMatrix=['Valid_Ascii_' WhichDataType];
ValidMatrix=eval(ValidMatrix);
TestMatrix=['Test_Ascii_' WhichDataType];
TestMatrix=eval(TestMatrix);

if strcmp(WhichDataType,'Indicator')
    SizeEachStep=49;
else
    SizeEachStep=6;
end

NoOfBatches=20;
NoOfHistorySize=1;
NoOfInputVectorsTrain=floor(size(TrainMatrix,1)/NoOfBatches);
NoOfInputVectorsTest=floor(size(TestMatrix,1)/NoOfBatches);
NoOfInputVectorsValid=floor(size(ValidMatrix,1)/NoOfBatches);
trainX=zeros(NoOfInputVectorsTrain-NoOfHistorySize,SizeEachStep*NoOfHistorySize);
trainY=zeros(NoOfInputVectorsTrain-NoOfHistorySize,1);
validX=zeros(NoOfInputVectorsValid-NoOfHistorySize,SizeEachStep*NoOfHistorySize);
validY=zeros(NoOfInputVectorsValid-NoOfHistorySize,1);
testX=zeros(NoOfInputVectorsTest-NoOfHistorySize,SizeEachStep*NoOfHistorySize);
testY=zeros(NoOfInputVectorsTest-NoOfHistorySize,1);

for i=1:1:NoOfBatches
    TempTrainX=TrainMatrix((i-1)*NoOfInputVectorsTrain+1:i*NoOfInputVectorsTrain,:);
    for j=NoOfHistorySize+1:1:NoOfInputVectorsTrain
        InputVec=(TempTrainX(j-NoOfHistorySize:j-1,:))';
        trainX(j-NoOfHistorySize,:)=InputVec(:);
        if strcmp(WhichDataType,'Indicator')
            trainY(j-NoOfHistorySize)=find(TempTrainX(j,:)==1);
        else
            trainY(j-NoOfHistorySize)=bi2de(flip(TempTrainX(j,:)));
        end
    end
    
   
    TempValidX=ValidMatrix((i-1)*NoOfInputVectorsValid+1:i*NoOfInputVectorsValid,:);
    for j=NoOfHistorySize+1:1:NoOfInputVectorsValid
        InputVec=(TempValidX(j-NoOfHistorySize:j-1,:))';
        validX(j-NoOfHistorySize,:)=InputVec(:);
        if strcmp(WhichDataType,'Indicator')
            validY(j-NoOfHistorySize)=find(TempValidX(j,:)==1);
        else
            validY(j-NoOfHistorySize)=bi2de(flip(TempValidX(j,:)));
        end

    end
    
    TempTestX=TestMatrix((i-1)*NoOfInputVectorsTest+1:i*NoOfInputVectorsTest,:);
    for j=NoOfHistorySize+1:1:NoOfInputVectorsTest
        InputVec=(TempTestX(j-NoOfHistorySize:j-1,:))';
        testX(j-NoOfHistorySize,:)=InputVec(:);
        if strcmp(WhichDataType,'Indicator')
            testY(j-NoOfHistorySize)=find(TempTestX(j,:)==1);
        else
            testY(j-NoOfHistorySize)=bi2de(flip(TempTestX(j,:)));
        end

    end
    
    save(strcat(PennTreebank_Character_DataPath,'_Batch',int2str(i)),'trainX','trainY','testX','testY','validX','validY','UniqueLabels','i','-v7.3')
end

