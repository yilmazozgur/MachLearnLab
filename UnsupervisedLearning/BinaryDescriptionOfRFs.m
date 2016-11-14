function ReceptiveFields=BinaryDescriptionOfRFs(Dataset, RFs, NumberOfRFs, optionsAll)
optionsKmeans=optionsAll{1,8};
optionsData = optionsAll{1,1};
if not(optionsKmeans.BinaryDescription==1)
    ReceptiveFields = RFs;
    return;
end

OrgImageDim = Dataset.ImageDim;
rfSize = optionsAll{1,8}.rfSize; 
Dataset.ImageDim = [rfSize rfSize 3];
RFs_=RFs.RFs;

% Sampling the RFs with BRIEF descriptor is implemted here...
ReceptiveFields_ = SamplingFrom_Spatial2(Dataset, RFs_, optionsAll);
if optionsData.Verbose 
    fprintf('Receptive Fields are sampled with BRIEF descriptor!\n'); 
end

Dataset.ImageDim = OrgImageDim;

ReceptiveFields = RFs;
ReceptiveFields.RFs = ReceptiveFields_;
end