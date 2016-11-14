function Features=BriefSubImages(Dataset,ReceptiveFields,DataStats,options) %optionsAll
tic
RootPath=options{10};
WhichData=options{1}.TrainTest;

if strcmp(WhichData,'Features') ||  strcmp(WhichData,'ExpandedFeatures') || strcmp(WhichData,'Model')
    Features=Dataset;
    return
end

%BatchFile='datasetname; ColorType; SummationType; NumberOfBitsPerSubimage; 
% SmoothingKernelSize; SD; ExtractionType; BatchNumber';
savedir=strcat(RootPath,'/BRIEF/BriefSubImages/Saved BRIEF Features/');
if options{9}.SmoothingFlag
    BatchFile=strcat(savedir,int2str(options{1}.TotalNumberOfBatches),';',Dataset.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_',options{3}.SummationType,'_{',...
    int2str(options{9}.NumberOfTests),'}_[',int2str(options{9}.SmoothKernel),']_',...
    num2str(options{9}.SD_SmoothingFilter),'_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
else
    BatchFile=strcat(savedir,int2str(options{1}.TotalNumberOfBatches),';',Dataset.DatasetName,'_',options{9}.Dimension,'_',options{9}.ColorType,'_',options{3}.SummationType,'_{',...
    int2str(options{9}.NumberOfTests),'}_[NO Smoothing]_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat');
end

% if exist(BatchFile,'file')
% load(BatchFile);
% if options{1}.Verbose
%     fprintf('Feature are loaded! :  %f\n',toc)
% end
%     return
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
optionsData      = options{1};                                           %
optionsExtract   = options{2};                   
optionsBrief     = options{9};                                           %
ColorType        = optionsBrief.ColorType;                             %
optionsPool      = options{3};                                           % 
optionsNormalize = options{4};                                           %
SubImgSize       = optionsPool.SummationType;                            %
kernel           = optionsBrief.SmoothKernel;                          %            
SD               = optionsBrief.SD_SmoothingFilter;                    %
Number_Of_Tests  = optionsBrief.NumberOfTests;                         %
Dimension        = optionsBrief.Dimension;    
ImageDim         = Dataset.ImageDim;                    % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if strcmp(optionsData.WhichData,'Features') ||  strcmp(optionsData.WhichData,'ExpandedFeatures') || strcmp(optionsData.WhichData,'Model')
    Features=Dataset;
    return
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%if no expansion is needed, return.                                             %
%This is necessary not to make a major change in the experiment script.         %
if not(optionsExtract.Perform)                                                  %
    if strcmp(Dataset.options{1,1}.TrainTest,'Train')                           %
        Features.Features=Dataset.trainX;                                       %
        Features.Labels=Dataset.trainY;                                         %
    else                                                                        %
        Features.Features=Dataset.testX;                                        %
        Features.Labels=Dataset.testY;                                          %
    end                                                                         %
    Features.UniqueLabels=Dataset.UniqueLabels;                                 %
    Features.optionsAll=options;                                                %
    Features.optionsPrev=options;                                               %    
    Features.RFData=ReceptiveFields;                                            %
    Features.SparsifyFlag=optionsExtract.SparsifyFlag;                          %
    Features.SparsityMultiplier=optionsExtract.SparsityMultiplier;              %
    Features.SummationType=optionsPool.SummationType;                           %
    Features.Binarize=optionsNormalize.Binarize;                                %
    Features.DatasetName=Dataset.DatasetName;                                   %
                                                                                %
    if strcmp(optionsData.TrainTest,'Train')                                    %
        if isempty(DataStats)                                                   %
            XPooled_mean = mean(Features.Features);                             %
            XPooled_sd = sqrt(var(Features.Features)+0.01);                     %
        else %for batch based processing, combine with previous batch stats     %
            XPooled_mean = (mean(Features.Features)+DataStats.Mean)/2;          %
            XPooled_sd = (sqrt(var(Features.Features)+0.01)+DataStats.Std)/2;   %
        end                                                                     %
    else                                                                        %
        XPooled_mean=DataStats.Mean;                                            %
        XPooled_sd=DataStats.Std;                                               %
    end                                                                         %
                                                                                %
    Features.Mean=XPooled_mean;                                                 %
    Features.Std=XPooled_sd;                                                    %
                                                                                %
    return                                                                      %
end                                                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X=Dataset.trainX;                                                             %
if strcmp(optionsData.TrainTest,'Train') %either Train/Test data is processed.  %
    X=Dataset.trainX; %data                                                     %
    Features.Labels=Dataset.trainY;                                             %
else                                                                            %
    X=Dataset.testX; %data                                                      %
    Features.Labels=Dataset.testY;                                              %
end                                                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
I=X(1,:);
I=reshape(I,ImageDim);
[m,n,~]=size(I);

                switch SubImgSize
                    case 'Single'
                        subm=m;   subn=n;      
                        Number_Of_Keypoints=1;            
                        cx=round(((subm/2):subm:m)); % center of subimages in x coordinates
                        cy=round(((subn/2):subn:n)); % center of subimages in y coordinates

                    case 'Coarse'
                        subm=round(m/2); subn=round(n/2);    
                        Number_Of_Keypoints=4;
                        cx=round(((subm/2):subm:m));       % center of subimages in x coordinates
                        cy=round(((subn/2):subn:n));       % center of subimages in y coordinates

                    case 'Fine'
                        subm=round(m/4); subn=round(n/4); 
                        Number_Of_Keypoints=16;
                        cx=round(((subm/2):subm:m)); 
                        cy=round(((subn/2):subn:n));

                    case 'Finest'
                        subm=round(m/8); subn=round(n/8);
                        Number_Of_Keypoints=64;
                        cx=round(((subm/2):subm:m)); 
                        cy=round(((subn/2):subn:n));
                end


if strcmp(Dimension,'2D')
    qq=zeros(size(X,1),Number_Of_Keypoints*sum(Number_Of_Tests));
    q=zeros(1,sum(Number_Of_Tests)*Number_Of_Keypoints);
elseif strcmp(Dimension,'3D')
    qq=zeros(size(X,1),Number_Of_Keypoints*Number_Of_Tests(1));
    q=zeros(1,Number_Of_Tests(1)*Number_Of_Keypoints);
end


h=fspecial('gaussian',kernel,SD);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %%%%%%%%%%%%%%%%%%%%%%
% put all x and y coors. of all centre pnts of subimages, 4-by-2 for 'Coarse'; 16-by-2 for 'Fine'; 
c_xy=zeros(Number_Of_Keypoints,2);
for p=1:length(cx)                                       
    for l=1:length(cy)                                   
            c_xy(length(cy)*(p-1)+l,:)=[cx(p),cy(l)];    
    end                                                                     
end 


PathPrefix=strcat(RootPath,'/BRIEF/BriefSubImages');
FileName=fullfile(PathPrefix,strcat('PD_',Dimension,'_(',int2str(Number_Of_Tests(1)),'*',int2str(Number_Of_Tests(2)),'*',int2str(Number_Of_Tests(3)),')tests_[',int2str(subm),'x',int2str(subn),']size','.mat'));

                       if exist(FileName,'file')
                           load(FileName);      

                       else
                           [x1,x2,y1,y2,z1,z2]=distribution(subm,Number_Of_Tests,FileName,Dimension);

                       end


x1_ch1=x1(1:Number_Of_Tests(1));
x1_ch2=x1(1:Number_Of_Tests(2));
x1_ch3=x1(1:Number_Of_Tests(3));

x2_ch1=x2(1:Number_Of_Tests(1));
x2_ch2=x2(1:Number_Of_Tests(2));
x2_ch3=x2(1:Number_Of_Tests(3));

y1_ch1=y1(1:Number_Of_Tests(1));
y1_ch2=y1(1:Number_Of_Tests(2));
y1_ch3=y1(1:Number_Of_Tests(3));

y2_ch1=y2(1:Number_Of_Tests(1));
y2_ch2=y2(1:Number_Of_Tests(2));
y2_ch3=y2(1:Number_Of_Tests(3));                      
                       
switch ColorType
    case 'RGB'
        R_=zeros(1,Number_Of_Tests(1));  
        G_=zeros(1,Number_Of_Tests(2));  
        B_=zeros(1,Number_Of_Tests(3));               
        RGB_=zeros(1,Number_Of_Tests(1));    
        for img=1:size(X,1)

                    I=X(img,:);
                    I=reshape(I,ImageDim);
                    if optionsBrief.SmoothingFlag
                        I=imfilter(I,h,'replicate');
                    end

                    
                     for j=1:Number_Of_Keypoints
                        ICh1=I(:,:,1);
                        ICh2=I(:,:,2);
                        ICh3=I(:,:,3);

                        
                        Coor1_1=max(c_xy(j,1)+x1_ch1-1,1);
                        Coor1_2=max(c_xy(j,2)+y1_ch1-1,1);
                        Coor1_3=max(c_xy(j,1)+x2_ch1-1,1);
                        Coor1_4=max(c_xy(j,2)+y2_ch1-1,1);

                        Coor2_1=max(c_xy(j,1)+x1_ch2-1,1);
                        Coor2_2=max(c_xy(j,2)+y1_ch2-1,1);
                        Coor2_3=max(c_xy(j,1)+x2_ch2-1,1);
                        Coor2_4=max(c_xy(j,2)+y2_ch2-1,1);

                        Coor3_1=max(c_xy(j,1)+x1_ch3-1,1);
                        Coor3_2=max(c_xy(j,2)+y1_ch3-1,1);
                        Coor3_3=max(c_xy(j,1)+x2_ch3-1,1);
                        Coor3_4=max(c_xy(j,2)+y2_ch3-1,1);
                        
                        if strcmp(Dimension,'3D')
                            CoorZ1=z1;
                            CoorZ2=z2;
                            Ind1_1=sub2ind_Lab_3D(ImageDim(1:3),Coor1_1,Coor1_2,CoorZ1);
                            Ind1_2=sub2ind_Lab_3D(ImageDim(1:3),Coor1_3,Coor1_4,CoorZ2);
                            
                            I1_1=I(Ind1_1);
                            I1_2=I(Ind1_2); 
                            RGB_(:)=not(I1_1 < I1_2); RGB_=double(RGB_(:));
                            q( Number_Of_Tests(1)*(j-1)+1 : Number_Of_Tests(1)*j )  = RGB_;
                            
                            
                        elseif strcmp(Dimension,'2D')
                            Ind1_1=sub2ind_Lab(ImageDim(1:2),Coor1_1,Coor1_2);
                            Ind1_2=sub2ind_Lab(ImageDim(1:2),Coor1_3,Coor1_4);
                            Ind2_1=sub2ind_Lab(ImageDim(1:2),Coor2_1,Coor2_2);
                            Ind2_2=sub2ind_Lab(ImageDim(1:2),Coor2_3,Coor2_4);
                            Ind3_1=sub2ind_Lab(ImageDim(1:2),Coor3_1,Coor3_2);
                            Ind3_2=sub2ind_Lab(ImageDim(1:2),Coor3_3,Coor3_4);
                       
                            I1_1=ICh1(Ind1_1);
                            I1_2=ICh1(Ind1_2);                        
                            I2_1=ICh2(Ind2_1);
                            I2_2=ICh2(Ind2_2);                        
                            I3_1=ICh3(Ind3_1);
                            I3_2=ICh3(Ind3_2);
                                                                                          
                            R_(:)=not(I1_1 < I1_2); R_=double(R_(:));
                            G_(:)=not(I2_1 < I2_2); G_=double(G_(:));
                            B_(:)=not(I3_1 < I3_2); B_=double(B_(:));
                        
                            q(sum(Number_Of_Tests)*(j-1)+1                        : sum(Number_Of_Tests)*(j-1)+Number_Of_Tests(1))   = R_;
                            q(sum(Number_Of_Tests)*(j-1)+Number_Of_Tests(1)+1     : sum(Number_Of_Tests)*(j-1)+(Number_Of_Tests(1) + Number_Of_Tests(2))) = G_;
                            q(sum(Number_Of_Tests)*(j-1)+(Number_Of_Tests(1) + Number_Of_Tests(2))+1 : sum(Number_Of_Tests)*(j-1)+sum(Number_Of_Tests)) = B_;
                        end
    
                     end        
                     qq(img,:)=q(:);
        end
        
    case 'Gray'
        Gray=zeros(1,Number_Of_Tests(1));
        clear x1_ch2 x1_ch3 x2_ch2 x2_ch3 y1_ch2 y1_ch3 y2_ch2 y2_ch3;
        for img=1:size(X,1)
                    I=X(img,:);
                    I=reshape(I,ImageDim);
                    
                    if optionsBrief.SmoothingFlag
                        I=imfilter(I,h,'replicate');
                    end
                    I=uint8(I);
                    I=rgb2gray(I);
                for j=1:Number_Of_Keypoints
                    
                        ICh=I(:,:);

                        Coor1_1=max(c_xy(j,1)+x1_ch1-1,1);
                        Coor1_2=max(c_xy(j,2)+y1_ch1-1,1);
                        Coor1_3=max(c_xy(j,1)+x2_ch1-1,1);
                        Coor1_4=max(c_xy(j,2)+y2_ch1-1,1);

                        Ind1_1=sub2ind_Lab([32.0 32.0],Coor1_1,Coor1_2);
                        Ind1_2=sub2ind_Lab([32.0 32.0],Coor1_3,Coor1_4);                       

                        I1_1=ICh(Ind1_1);
                        I1_2=ICh(Ind1_2);                                               
                       
                        Gray(:)=not(I1_1 < I1_2); Gray=double(Gray(:));
                        
                    q( Number_Of_Tests(1)*(j-1)+1 : Number_Of_Tests(1)*j )  = Gray;
                   
                end
                   qq(img,:)=q(:);
        end;
        
        
    case 'YCbCr'
        Y=zeros(1,Number_Of_Tests(1));   
        Cb=zeros(1,Number_Of_Tests(2));   
        Cr=zeros(1,Number_Of_Tests(3));
        YCbCr_=zeros(1,Number_Of_Tests(1));
        for img=1:size(X,1)

                    I=X(img,:);
                    I=reshape(I,[32 32 3]);
                    
                    if optionsBrief.SmoothingFlag
                        I=imfilter(I,h,'replicate');
                    end
                    I=rgb2ycbcr(I);
                for j=1:Number_Of_Keypoints
                        ICh1=I(:,:,1);
                        ICh2=I(:,:,2);
                        ICh3=I(:,:,3);
                       
                        Coor1_1=max(c_xy(j,1)+x1_ch1-1,1);
                        Coor1_2=max(c_xy(j,2)+y1_ch1-1,1);
                        Coor1_3=max(c_xy(j,1)+x2_ch1-1,1);
                        Coor1_4=max(c_xy(j,2)+y2_ch1-1,1);
                        Coor2_1=max(c_xy(j,1)+x1_ch2-1,1);
                        Coor2_2=max(c_xy(j,2)+y1_ch2-1,1);
                        Coor2_3=max(c_xy(j,1)+x2_ch2-1,1);
                        Coor2_4=max(c_xy(j,2)+y2_ch2-1,1);
                        Coor3_1=max(c_xy(j,1)+x1_ch3-1,1);
                        Coor3_2=max(c_xy(j,2)+y1_ch3-1,1);
                        Coor3_3=max(c_xy(j,1)+x2_ch3-1,1);
                        Coor3_4=max(c_xy(j,2)+y2_ch3-1,1);
                        
                       if strcmp(Dimension,'3D')
                            CoorZ1=z(1:end/2);
                            CoorZ2=z(end/2+1:end);
                            Ind1_1=sub2ind(ImageDim(1:3),Coor1_1,Coor1_2,CoorZ1);
                            Ind1_2=sub2ind(ImageDim(1:3),Coor1_3,Coor1_4,CoorZ2);
                            
                            I1_1=I(Ind1_1);
                            I1_2=I(Ind1_2); 
                            YCbCr_(:)=not(I1_1 < I1_2); YCbCr_=double(YCbCr_(:));
                            q( Number_Of_Tests(1)*(j-1)+1 : Number_Of_Tests(1)*j )  = YCbCr_;
                            
                            
                       else strcmp(Dimension,'2D')                       
                            Ind1_1=sub2ind_Lab(ImageDim(1:2),Coor1_1,Coor1_2);
                            Ind1_2=sub2ind_Lab(ImageDim(1:2),Coor1_3,Coor1_4);
                            Ind2_1=sub2ind_Lab(ImageDim(1:2),Coor2_1,Coor2_2);
                            Ind2_2=sub2ind_Lab(ImageDim(1:2),Coor2_3,Coor2_4);
                            Ind3_1=sub2ind_Lab(ImageDim(1:2),Coor3_1,Coor3_2);
                            Ind3_2=sub2ind_Lab(ImageDim(1:2),Coor3_3,Coor3_4);

                            I1_1=ICh1(Ind1_1);
                            I1_2=ICh1(Ind1_2);
                            I2_1=ICh2(Ind2_1);
                            I2_2=ICh2(Ind2_2);
                            I3_1=ICh3(Ind3_1);
                            I3_2=ICh3(Ind3_2);
                                                
                            Y(:)=not(I1_1 < I1_2); Y=double(Y(:));
                            Cb(:)=not(I2_1 < I2_2); Cb=double(Cb(:));
                            Cr(:)=not(I3_1 < I3_2); Cr=double(Cr(:));
                        
                            q(sum(Number_Of_Tests)*(j-1)+1                        : sum(Number_Of_Tests)*(j-1)+Number_Of_Tests(1))   = Y;
                            q(sum(Number_Of_Tests)*(j-1)+Number_Of_Tests(1)+1     : sum(Number_Of_Tests)*(j-1)+(Number_Of_Tests(1) + Number_Of_Tests(2))) = Cb;
                            q(sum(Number_Of_Tests)*(j-1)+(Number_Of_Tests(1) + Number_Of_Tests(2))+1 : sum(Number_Of_Tests)*(j-1)+sum(Number_Of_Tests)) = Cr;
                       end
                end
                   qq(img,:)=q(:);
        end;
        
        
    case 'HSV'
         H=zeros(1,Number_Of_Tests(1));   
         S=zeros(1,Number_Of_Tests(2));   
         V=zeros(1,Number_Of_Tests(3));
         HSV_=zeros(1,Number_Of_Tests(1));
        
        for img=1:size(X,1)

                    I=X(img,:);
                    I=reshape(I,[32 32 3]);
                    
                    if optionsBrief.SmoothingFlag
                        I=imfilter(I,h,'replicate');
                    end
                    I=rgb2hsv(I);
                for j=1:Number_Of_Keypoints
                        ICh1=I(:,:,1);
                        ICh2=I(:,:,2);
                        ICh3=I(:,:,3);
                       
                        Coor1_1=max(c_xy(j,1)+x1_ch1-1,1);
                        Coor1_2=max(c_xy(j,2)+y1_ch1-1,1);
                        Coor1_3=max(c_xy(j,1)+x2_ch1-1,1);
                        Coor1_4=max(c_xy(j,2)+y2_ch1-1,1);

                        Coor2_1=max(c_xy(j,1)+x1_ch2-1,1);
                        Coor2_2=max(c_xy(j,2)+y1_ch2-1,1);
                        Coor2_3=max(c_xy(j,1)+x2_ch2-1,1);
                        Coor2_4=max(c_xy(j,2)+y2_ch2-1,1);

                        Coor3_1=max(c_xy(j,1)+x1_ch3-1,1);
                        Coor3_2=max(c_xy(j,2)+y1_ch3-1,1);
                        Coor3_3=max(c_xy(j,1)+x2_ch3-1,1);
                        Coor3_4=max(c_xy(j,2)+y2_ch3-1,1);
                   if strcmp(Dimension,'3D')
                            CoorZ1=z(1:end/2);
                            CoorZ2=z(end/2+1:end);
                            Ind1_1=sub2ind(ImageDim(1:3),Coor1_1,Coor1_2,CoorZ1);
                            Ind1_2=sub2ind(ImageDim(1:3),Coor1_3,Coor1_4,CoorZ2);
                            
                            I1_1=I(Ind1_1);
                            I1_2=I(Ind1_2); 
                            HSV_(:)=not(I1_1 < I1_2); HSV_=double(HSV_(:));
                            q( Number_Of_Tests(1)*(j-1)+1 : Number_Of_Tests(1)*j )  = HSV_;
                            
                   elseif strcmp(Dimension,'2D') 
                        Ind1_1=sub2ind_Lab([32.0 32.0],Coor1_1,Coor1_2);
                        Ind1_2=sub2ind_Lab([32.0 32.0],Coor1_3,Coor1_4);
                        Ind2_1=sub2ind_Lab([32.0 32.0],Coor2_1,Coor2_2);
                        Ind2_2=sub2ind_Lab([32.0 32.0],Coor2_3,Coor2_4);
                        Ind3_1=sub2ind_Lab([32.0 32.0],Coor3_1,Coor3_2);
                        Ind3_2=sub2ind_Lab([32.0 32.0],Coor3_3,Coor3_4);

                        I1_1=ICh1(Ind1_1);
                        I1_2=ICh1(Ind1_2);                        
                        I2_1=ICh2(Ind2_1);
                        I2_2=ICh2(Ind2_2);                        
                        I3_1=ICh3(Ind3_1);
                        I3_2=ICh3(Ind3_2);                       
                        
                        H(:)=not(I1_1 < I1_2); H=double(H(:));
                        S(:)=not(I2_1 < I2_2); S=double(S(:));
                        V(:)=not(I3_1 < I3_2); V=double(V(:));
                        
                    q(sum(Number_Of_Tests)*(j-1)+1                        : sum(Number_Of_Tests)*(j-1)+Number_Of_Tests(1))   = H;
                    q(sum(Number_Of_Tests)*(j-1)+Number_Of_Tests(1)+1     : sum(Number_Of_Tests)*(j-1)+(Number_Of_Tests(1) + Number_Of_Tests(2))) = S;
                    q(sum(Number_Of_Tests)*(j-1)+(Number_Of_Tests(1) + Number_Of_Tests(2))+1 : sum(Number_Of_Tests)*(j-1)+sum(Number_Of_Tests)) = V;
                   end
                end
                   qq(img,:)=q(:);
        end
    case 'YIQ'
        Y=zeros(1,Number_Of_Tests(1));
        II=zeros(1,Number_Of_Tests(2));
        Q=zeros(1,Number_Of_Tests(3));
        YIQ_=zeros(1,Number_Of_Tests(1));
        for img=1:size(X,1)

                    I=X(img,:);
                    I=reshape(I,[32 32 3]);
                    
                    if optionsBrief.SmoothingFlag
                        I=imfilter(I,h,'replicate');
                    end
                    I=rgb2ntsc(I);
                for j=1:Number_Of_Keypoints
                        ICh1=I(:,:,1);
                        ICh2=I(:,:,2);
                        ICh3=I(:,:,3);
                       
                        Coor1_1=max(c_xy(j,1)+x1_ch1-1,1);
                        Coor1_2=max(c_xy(j,2)+y1_ch1-1,1);
                        Coor1_3=max(c_xy(j,1)+x2_ch1-1,1);
                        Coor1_4=max(c_xy(j,2)+y2_ch1-1,1);

                        Coor2_1=max(c_xy(j,1)+x1_ch2-1,1);
                        Coor2_2=max(c_xy(j,2)+y1_ch2-1,1);
                        Coor2_3=max(c_xy(j,1)+x2_ch2-1,1);
                        Coor2_4=max(c_xy(j,2)+y2_ch2-1,1);

                        Coor3_1=max(c_xy(j,1)+x1_ch3-1,1);
                        Coor3_2=max(c_xy(j,2)+y1_ch3-1,1);
                        Coor3_3=max(c_xy(j,1)+x2_ch3-1,1);
                        Coor3_4=max(c_xy(j,2)+y2_ch3-1,1);
                       
                   if strcmp(Dimension,'3D')
                            CoorZ1=z(1:end/2);
                            CoorZ2=z(end/2+1:end);
                            Ind1_1=sub2ind(ImageDim(1:3),Coor1_1,Coor1_2,CoorZ1);
                            Ind1_2=sub2ind(ImageDim(1:3),Coor1_3,Coor1_4,CoorZ2);
                            
                            I1_1=I(Ind1_1);
                            I1_2=I(Ind1_2); 
                            YIQ_(:)=not(I1_1 < I1_2); YIQ_=double(YIQ_(:));
                            q( Number_Of_Tests(1)*(j-1)+1 : Number_Of_Tests(1)*j )  = YIQ_;
                            
                            
                   elseif strcmp(Dimension,'2D')                        
                        Ind1_1=sub2ind_Lab([32.0 32.0],Coor1_1,Coor1_2);
                        Ind1_2=sub2ind_Lab([32.0 32.0],Coor1_3,Coor1_4);
                        Ind2_1=sub2ind_Lab([32.0 32.0],Coor2_1,Coor2_2);
                        Ind2_2=sub2ind_Lab([32.0 32.0],Coor2_3,Coor2_4);
                        Ind3_1=sub2ind_Lab([32.0 32.0],Coor3_1,Coor3_2);
                        Ind3_2=sub2ind_Lab([32.0 32.0],Coor3_3,Coor3_4);

                        I1_1=ICh1(Ind1_1);
                        I1_2=ICh1(Ind1_2);                        
                        I2_1=ICh2(Ind2_1);
                        I2_2=ICh2(Ind2_2);
                        I3_1=ICh3(Ind3_1);
                        I3_2=ICh3(Ind3_2);
                        
                        Y(:)=not(I1_1 < I1_2); Y=double(Y(:));
                        II(:)=not(I2_1 < I2_2); II=double(II(:));
                        Q(:)=not(I3_1 < I3_2); Q=double(Q(:));
                        
                        q(sum(Number_Of_Tests)*(j-1)+1                        : sum(Number_Of_Tests)*(j-1)+Number_Of_Tests(1))   = Y;
                        q(sum(Number_Of_Tests)*(j-1)+Number_Of_Tests(1)+1     : sum(Number_Of_Tests)*(j-1)+(Number_Of_Tests(1) + Number_Of_Tests(2))) = II;
                        q(sum(Number_Of_Tests)*(j-1)+(Number_Of_Tests(1) + Number_Of_Tests(2))+1 : sum(Number_Of_Tests)*(j-1)+sum(Number_Of_Tests)) = Q;
                   end
                end
                   qq(img,:)=q(:);
        end
end
        

if strcmp(optionsData.TrainTest,'Train')
    if isempty(DataStats)
        XPooled_mean = mean(qq);
        XPooled_sd = sqrt(var(qq)+0.01);
    else %for batch based processing, combine with previous batch stats
        XPooled_mean = (mean(qq)+DataStats.Mean)/2;
        XPooled_sd = (sqrt(var(qq)+0.01)+DataStats.Std)/2;
    end
else
    XPooled_mean=DataStats.Mean;
    XPooled_sd=DataStats.Std;
end

Features.Features=qq;
Features.Mean=XPooled_mean;
Features.Std=XPooled_sd;
Features.optionsAll=options;
Features.optionsPrev=options;
Features.RFData=ReceptiveFields;
Features.SparsifyFlag=optionsExtract.SparsifyFlag;
Features.Binarize=optionsNormalize.Binarize;
Features.SparsityMultiplier=optionsExtract.SparsityMultiplier;
Features.SummationType=optionsPool.SummationType;
Features.DatasetName=Dataset.DatasetName;
Features.UniqueLabels=[1,2,3,4,5,6,7,8,9,10];
if optionsData.Verbose
    fprintf('Feature Extraction done! :  %f\n',toc)
end
    if optionsBrief.SmoothingFlag
        save(strcat(savedir,int2str(optionsData.TotalNumberOfBatches),';',Dataset.DatasetName,'_',Dimension,'_',ColorType,'_',SubImgSize,'_{',int2str(Number_Of_Tests),'}_[',int2str(kernel),']_',num2str(SD),'_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat'),'Features','-v7.3');
    else
        save(strcat(savedir,int2str(optionsData.TotalNumberOfBatches),';',Dataset.DatasetName,'_',Dimension,'_',ColorType,'_',SubImgSize,'_{',int2str(Number_Of_Tests),'}_[NO Smoothing]_BRIEF_Batch_',int2str(options{1,1}.BatchNumber),'.mat'),'Features','-v7.3');
    end
end
