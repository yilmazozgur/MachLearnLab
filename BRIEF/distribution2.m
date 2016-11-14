function [x1,x2,y1,y2,z1,z2]=distribution2(subm,Number_Of_Tests,FileName,Dimension)
fprintf('_________________________________***************************__________________________________________________\n');
            S=floor(subm/2); 

%                     z=ceil(rand(Number_Of_Tests(1)*2,1)*3);
                        % z(k)=ceil(rand*3);
%                     xy_pair=round(((rand(max(Number_Of_Tests)*4,1)*2)-1)*(S-1));
% %      
%                       x1 = xy_pair(1:max(Number_Of_Tests));
%                       y1 = xy_pair(max(Number_Of_Tests)+1:max(Number_Of_Tests)*2);
%                       x2 = xy_pair(max(Number_Of_Tests)*2+1:max(Number_Of_Tests)*3);
%                       y2 = xy_pair(max(Number_Of_Tests)*3+1:end);
                 cn=0;
                 for x1_=-S+1:1:S
                     for y1_=-S+1:1:S
                        for x2_=-S+1:1:S
                            for y2_=-S+1:1:S
                               if strcmp(Dimension,'3D')
                                    for z1_=1:3
                                        for z2_=1:3
                                            cn=cn+1;
                                            xyz_pair_(cn,:)=[x1_ y1_ x2_ y2_ z1_ z2_];
                                        end
                                    end
                               else
                                    cn=cn+1;
                                    xy_pair_(cn,:)=[x1_ y1_ x2_ y2_];
                               end  
                            end
                        end
                     end
                 end
                 
                 if strcmp(Dimension,'3D')
                     len=length(xyz_pair_);
                     xyz_pair=xyz_pair_(randperm(len,max(Number_Of_Tests)),1:6);
                     x1=xyz_pair(:,1);
                     y1=xyz_pair(:,2);
                     x2=xyz_pair(:,3);
                     y2=xyz_pair(:,4);
                     z1=xyz_pair(1:Number_Of_Tests(1),5); 
                     z2=xyz_pair(1:Number_Of_Tests(1),6);
                 else
                     len=length(xy_pair_);
                     xy_pair=xy_pair_(randperm(len,max(Number_Of_Tests)),1:4);
                     x1=xy_pair(:,1);
                     y1=xy_pair(:,2);
                     x2=xy_pair(:,3);
                     y2=xy_pair(:,4);
                 end

                 
                 
                 
                 count=0;
                 for i=1:max(Number_Of_Tests)
                     count_=0;
                     for j=2:i-1
                         if (x1(i)==x1(j)) && (x2(i)==x2(j)) && (y1(i)==y1(j)) && (y2(i)==y2(j)) && (z1(i)==z1(j)) && (z2(i)==z2(j))
%                              if (x1(i)==x2(j)) && (y1(i)==y2(j)) && (z1(i)==z2(j))
                                 count_=count_+1;
                         end
                     end
                     if count_>=1 
                         count=count+1;
                     end
                 end
                 

%                       z=z(end/2+1:end); 
                    save(FileName,'x1','x2','y1','y2','z1','z2');
                        
fprintf('=============================%d=========================',count);
            
end