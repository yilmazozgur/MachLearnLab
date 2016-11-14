function [qq]=SamplingFrom_Fourier(Dataset,X,options)
PatchSize=options{9}.PatchSize1;
stride=options{9}.Stride1;
Number_Of_Tests=options{9}.NumberOfTests1;
RootPath=options{10};

ImageDim=Dataset.ImageDim;
ImageDim2=ImageDim(1:3);
k__= [1 cumprod(ImageDim2(1:end-1))];

optionsBrief     = options{9};  
ColorType        = optionsBrief.ColorType;                                     
kernel           = optionsBrief.SmoothKernel;                                     
SD               = optionsBrief.SD_SmoothingFilter;           
Dimension        = optionsBrief.Dimension;    


m=ImageDim(1);
n=ImageDim(2);

subm=PatchSize(1);
subn=PatchSize(2);
if (options{8}.BinaryDescription==1)
    Number_Of_Keypoints = 1;
    cx = floor(m/2);
    cy = floor(n/2);
else
    Number_Of_Keypoints=floor(((m-subm)/stride)+1)*floor(((n-subn)/stride)+1);
    cx=round(((subm/2):stride:m-subm/2)); % center of subimages in x coordinates
    cy=round(((subn/2):stride:n-subn/2)); % center of subimages in y coordinates
end

if strcmp(Dimension,'2D') && not(strcmp(ColorType,'Gray'))
    
    qq=zeros(size(X,1),Number_Of_Keypoints*sum(Number_Of_Tests));
        q=zeros(1,sum(Number_Of_Tests)*Number_Of_Keypoints);

elseif strcmp(Dimension,'3D') || strcmp(ColorType,'Gray')
    
    
    qq=zeros(size(X,1),Number_Of_Keypoints*Number_Of_Tests(1));
    q=zeros(1,Number_Of_Tests(1)*Number_Of_Keypoints);
    
end




h=fspecial('gaussian',kernel,SD);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% put all x and y coors. of all centre pnts of subimages, 4-by-2 for 'Coarse'; 16-by-2 for 'Fine'; 
c_xy=zeros(Number_Of_Keypoints,2);
for p=1:length(cx)                                       
    for k=1:length(cy)                                   
            c_xy(length(cy)*(p-1)+k,:)=[cx(p),cy(k)];    
    end                                                                     
end 


% PathPrefix=strcat(RootPath,'/BRIEF/BriefOverlappedPatches');
PathPrefix=strcat('F:\Recurrent Holistic Vision v0.4\SavedData\BriefOverlappedPatches');
FileName=fullfile(PathPrefix,strcat('PD_',Dimension,'_(',int2str(Number_Of_Tests(1)),'x',int2str(Number_Of_Tests(2)),'x',int2str(Number_Of_Tests(3)),')tests_[',int2str(subm),'x',int2str(subn),']size','.mat'));
FileName2=fullfile(PathPrefix,strcat('GA_',int2str(options{8}.BitsPerRFs),'_random','.mat'));

                       if exist(FileName,'file')
                           load(FileName);   
                           load(FileName2);  
                       else
                           if strcmp(Dimension,'3D')
                               [x1,x2,y1,y2,z1,z2]=distribution3(subm,Number_Of_Tests,FileName,Dimension);
                           elseif strcmp(Dimension,'2D')
                               [x1,x2,y1,y2]=distribution3(subm,Number_Of_Tests,FileName,Dimension);
                           end
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
%%%%%%%%%%%__RGB Sampling___%%%%%%%%%%%  
            R_=zeros(1,Number_Of_Tests(1));  
            G_=zeros(1,Number_Of_Tests(2));  
            B_=zeros(1,Number_Of_Tests(3));          
            if strcmp(Dimension,'3D')
                RGB_=zeros(1,Number_Of_Tests(1));  
            end;
            for img=1:size(X,1)
                        I=X(img,:);
                        Ivec=I;
                        I=reshape(I,ImageDim);
                        if optionsBrief.SmoothingFlag
                            I=imfilter(I,h,'replicate');
                        end


                         for j=1:Number_Of_Keypoints
                                ICh1=I(:,:,1);
                                ICh2=I(:,:,2);
                                ICh3=I(:,:,3);
                            
                           if img==1 && j==1
                                Coor1_1=min(max(c_xy(j,1)+x1_ch1,1),subm);
                                Coor1_2=min(max(c_xy(j,2)+y1_ch1,1),subm);
                                Coor1_3=min(max(c_xy(j,1)+x2_ch1,1),subm);
                                Coor1_4=min(max(c_xy(j,2)+y2_ch1,1),subm);

                                Coor2_1=min(max(c_xy(j,1)+x1_ch2,1),subm);
                                Coor2_2=min(max(c_xy(j,2)+y1_ch2,1),subm);
                                Coor2_3=min(max(c_xy(j,1)+x2_ch2,1),subm);
                                Coor2_4=min(max(c_xy(j,2)+y2_ch2,1),subm);

                                Coor3_1=min(max(c_xy(j,1)+x1_ch3,1),subm);
                                Coor3_2=min(max(c_xy(j,2)+y1_ch3,1),subm);
                                Coor3_3=min(max(c_xy(j,1)+x2_ch3,1),subm);
                                Coor3_4=min(max(c_xy(j,2)+y2_ch3,1),subm);
                           end
                            if strcmp(Dimension,'3D')
                                 if img==1 && j==1
                                    CoorZ1=z1;
                                    CoorZ2=z2;
                                    Abc=[Coor1_1,Coor1_2,CoorZ1];
                                    Abcd=[Coor1_3,Coor1_4,CoorZ2];
                                
                               
                                    Ind1_1=sub2ind_Lab_3D_2(ImageDim(1:3),k__,Abc);
                                    Ind1_2=sub2ind_Lab_3D_2(ImageDim(1:3),k__,Abcd);

                                end
%                                 I1_1=I(Ind1_1);
%                                 I1_2=I(Ind1_2);
                                I1_1=Ivec(Pattern(:,1));
                                I1_2=Ivec(Pattern(:,2));
                                

                                q(Number_Of_Tests(1)*(j-1)+1 : Number_Of_Tests(1)*j )=not(I1_1 > I1_2); 
%                                 q( Number_Of_Tests(1)*(j-1)+1 : Number_Of_Tests(1)*j )  = RGB_;               

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
                         qq(img,:)=q;
            end
            
            
            

    case 'Gray'
%%%%%%%%%%%__Gray Sampling___%%%%%%%%%%%
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
                    
                   if not(strcmp(options{9}.Domain,'Spatial'))
                        freqI=abs(fft2(I));
                   end
                   
                for j=1:Number_Of_Keypoints
                    
                        ICh=I(:,:);

                        Coor1_1=max(c_xy(j,1)+x1_ch1-1,1);
                        Coor1_2=max(c_xy(j,2)+y1_ch1-1,1);
                        Coor1_3=max(c_xy(j,1)+x2_ch1-1,1);
                        Coor1_4=max(c_xy(j,2)+y2_ch1-1,1);

                        Ind1_1=sub2ind_Lab([32.0 32.0],Coor1_1,Coor1_2);
                        Ind1_2=sub2ind_Lab([32.0 32.0],Coor1_3,Coor1_4);                       
 if strcmp(options{9}.Domain,'FrequencyMagnitude')
     freqMagnitude_=zeros(1,Number_Of_Tests(1));
     
     
       freqI1_1=freqI(Ind1_1);
       freqI1_2=freqI(Ind1_2);
       freqMagnitude_(:)=not(freqI1_1 < freqI1_2); 
       freqMagnitude_=double(freqMagnitude_(:));
       q( Number_Of_Tests(1)*(j-1)+1 : Number_Of_Tests(1)*j )  = freqMagnitude_;
 elseif strcmp(options{9}.Domain,'Spatial')
     I1_1=ICh(Ind1_1);
     I1_2=ICh(Ind1_2);    
     Gray(:)=not(I1_1 < I1_2); Gray=double(Gray(:));
     q( Number_Of_Tests(1)*(j-1)+1 : Number_Of_Tests(1)*j )  = Gray;
 end
                end
                   qq(img,:)=q(:);
            end
            
            
       
    case 'YCbCr'
%%%%%%%%%%%__YCbCr Sampling___%%%%%%%%%%% 
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
                            CoorZ1=z1;
                            CoorZ2=z2;
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
            end

            

    case 'HSV'
%%%%%%%%%%%__HSV Sampling___%%%%%%%%%%%                
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
                                CoorZ1=z1;
                                CoorZ2=z2;
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
%%%%%%%%%%%__YIQ Sampling___%%%%%%%%%%%  
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
                                CoorZ1=z1;
                                CoorZ2=z2;
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