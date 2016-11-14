%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function learns receptive fields from raw data for vision/machine
%learning tasks.
%
%%% Input:
%Dataset : A struct of the dataset created by LoadDataset.m
%NumberOfRFs: Number of receptive fields to be learned. Default=200.
%rfSize: The size of the RF. Default=6.
%options: struct for options. 
%options.ImageIsGrayFlag: flag for BW image. Default=false.
%options.GrayConvertFlag: flag to convert to BW. Default=false.
%options.numPatches: The number of randomly extracted patches.Default=10^6;
%options.ShowCentroids: show learned RFs. Default: true.
%options.SaveRFs: Save the results to the Root when finished.
%   Default=true.
%options.SavePath: the path to save the results if SaveRFs flag is up.
%   Default='';
%
%%% Output:
%ReceptiveFields: A struct, that has centroids of RFs, mean and covariance 
%matrices. See below for details.
%
%
%From:
%TOU_ML
%Ozgur Yilmaz, Turgut Ozal University, Ankara
%Web: ozguryilmazresearch.net
%Ref: It is based on Adam Coates code, Coates and Ng, 2011 paper.
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function ReceptiveFields=KmeansUnsupervised(Dataset,NumberOfRFs,rfSize,optionsAll)

%measure computation time
tic

options=optionsAll{8};
optionsData=optionsAll{1};


if options.AutoLoad
    FilePrefix='KmeansRF_';
    if optionsData.DatasetSequenceFlag %if sequence task, then skip this (by loading a pre-saved file)
        Dataset.Name='CIFAR10';
    end
    FileName=strcat(fullfile(options.SavePath,strcat(FilePrefix,Dataset.Name,'_NoRFs_',int2str(NumberOfRFs),'RFSize_',int2str(rfSize))),'.mat');
    if exist(FileName, 'file')
        load(FileName);
        if optionsData.Verbose 
            fprintf('Receptive Fields Loaded. \n'); 
        end
        return
    end
end

if optionsData.DatasetSequenceFlag %if sequence task, then skip this (by loading a pre-saved file)
    return
end

%default options, if not given by the user
if nargin < 4
    options.ImageIsGrayFlag=false;
    options.GrayConvertFlag=false;
    options.numPatches=10^6;
    options.ShowCentroids=true;
    options.SaveRFs=false;
    options.SavePath='';
    
    %default for RF size
    if nargin < 3
        rfSize=6;
        %default for Number of RFs
        if nargin < 2
            NumberOfRFs=200;
        end
    end
end

%extract the raw data and image dimensions
trainX=Dataset.trainX;
ImageDim=Dataset.ImageDim;

%if image is gray, or it is intended to be gray, initialize accordingly
if options.ImageIsGrayFlag || options.GrayConvertFlag
    patches = zeros(options.numPatches, rfSize*rfSize);
else
    if options.TwoChannelFlag
        patches = zeros(options.numPatches, rfSize*rfSize*2);
    else
        patches = zeros(options.numPatches, rfSize*rfSize*3);
    end
end

%Show what is to be done
if optionsData.Verbose 
    fprintf('K means Unsupervised Learning Start: \n'); 
end

%extract patched randomly for k means.
for i=1:options.numPatches
  if (mod(i,100000) == 0) && optionsData.Verbose 
      fprintf('Extracting patch: %d / %d\n', i, options.numPatches); 
  end

  r = random('unid', ImageDim(1) - rfSize + 1);
  c = random('unid', ImageDim(2) - rfSize + 1);
  patch = reshape(trainX(mod(i-1,size(trainX,1))+1, :), ImageDim);
  patch = patch(r:r+rfSize-1,c:c+rfSize-1,:);

  %if gray, then convert
  if options.ImageIsGrayFlag || options.GrayConvertFlag
      patch=double(rgb2gray(uint8(patch)));
  end

  patches(i,:) = patch(:)';
end

% normalize for contrast
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

% whiten
C = cov(patches);
M = mean(patches);
[V,D] = eig(C);
P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
patches = bsxfun(@minus, patches, M) * P;


% run K-means
centroids = run_kmeans(patches, NumberOfRFs, 100);

%show learned RFs
if options.ShowCentroids
    show_centroids_Kadir(centroids, rfSize); drawnow;
end

%form the struct to be saved
ReceptiveFields.options=options;
ReceptiveFields.RFs=centroids;
ReceptiveFields.RFSize=rfSize;
ReceptiveFields.M=M;
ReceptiveFields.P=P;
ReceptiveFields.DatasetName=Dataset.Name;
ReceptiveFields.ImageDim=Dataset.ImageDim;
ReceptiveFields.ImageIsGrayFlag=options.ImageIsGrayFlag;
ReceptiveFields.GrayConvertFlag=options.GrayConvertFlag;


%save if instructed
if options.SaveRFs
    save(fullfile(options.SavePath,strcat('KmeansRF_',Dataset.Name,'_NoRFs_',int2str(NumberOfRFs),'RFSize_',int2str(rfSize))),'ReceptiveFields')
    if optionsData.Verbose 
        fprintf('Receptive Fields saved. \n'); 
    end
end

%show the finalization on command window
if optionsData.Verbose 
    fprintf('K means learning done!:    %d \n', toc); 
end

end%end function


