
for i=1:1:5
    load('F:\Recurrent Holistic Vision v0.4\RawData\CIFAR10_YCbCr_SingleBatch\CIFAR10_YCbCr_Batch0.mat')
    trainX=trainX((i-1)*10000+1:i*10000,:);
    testX=testX((i-1)*2000+1:i*2000,:);
    trainY=trainY((i-1)*10000+1:i*10000);
    testY=testY((i-1)*2000+1:i*2000);
    save(strcat('F:\Recurrent Holistic Vision v0.4\RawData\CIFAR10_YCbCr_SingleBatch\CIFAR10_YCbCr_Batch',int2str(i),'.mat'))
end