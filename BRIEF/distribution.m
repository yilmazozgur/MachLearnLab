function [x1,x2,y1,y2,z]=distribution(subm,Number_Of_Tests,FileName,~)
fprintf('_________________________________***************************__________________________________________________\n');
            S=floor(subm/2); 
%             sigma=S/5;
%             pd=makedist('Normal','mu',0,'sigma',sigma);
%             

%                     xy_pair=zeros(max(Number_Of_Tests)*4,1);
%                     z=zeros(Number_Of_Tests(1),1);
                    z=ceil(rand(Number_Of_Tests(1)*2,1)*3); % z(k)=ceil(rand*3);

%                     for k=1:size(xy_pair,1)
%                         xy_pair(k)=round(random(pd));
%                     end

                        xy_pair=round(((rand(max(Number_Of_Tests)*4,1)*2)-1)*(S-1));
                        
%                         x1=xy_pair(1:4:end);
%                         x2=xy_pair(2:4:end);
%                         y1=xy_pair(3:4:end);
%                         y2=xy_pair(4:4:end);
                        
                        
                      x1 = xy_pair(1:max(Number_Of_Tests));
                      y1 = xy_pair(max(Number_Of_Tests)+1:max(Number_Of_Tests)*2);
                      x2 = xy_pair(max(Number_Of_Tests)*2+1:max(Number_Of_Tests)*3);
                      y2 = xy_pair(max(Number_Of_Tests)*3+1:end);
                      
                 for itrn=1:50
                     v=0;
                     for i=2:max(Number_Of_Tests)
                          for j=1:i-1
                              if (x1(i)==x1(j)) && (x2(i)==x2(j)) && (y1(i)==y1(j)) && (y2(i)==y2(j)) && (z(i)==z(j))
                                  v=v+1;
                                if mod(itrn,4)==0
                                      if ((y2(j)-1) < (-S+1)) 
                                            y2(j)=y2(j)+1;
                                      else
                                          y2(j)=y2(j)-1;
                                      end

                                elseif mod(itrn,4)==1
                                      if ((y1(j)-1) < (-S+1)) 
                                            y1(j)=y1(j)+1;
                                      else
                                          y1(j)=y1(j)-1;
                                      end 
                                elseif mod(itrn,4)==2
                                      if ((x2(j)-1) < (-S+1)) 
                                            x2(j)=x2(j)+1;
                                      else
                                          x2(j)=x2(j)-1;
                                      end
                                else 
                                      if ((x1(j)-1) < (-S+1)) 
                                            x1(j)=x1(j)+1;
                                      else
                                          x1(j)=x1(j)-1;
                                      end
                                end
                                  
%                                   continue;
                              end
                          end
                          
                     end
                     if v==0
                         break;
                     end
                 end







                 
                 count=0;
                 for i=2:max(Number_Of_Tests)
                     for j=1:i-1
                         if (x1(i)==x1(j)) && (x2(i)==x2(j)) && (y1(i)==y1(j)) && (y2(i)==y2(j)) && (z(i)==z(j))
                                 count=count+1;
                         end
                     end
                 end
                 

%                       z=z(end/2+1:end); 
                    save(FileName,'x1','x2','y1','y2','z');
                        
fprintf('=============================%d=========================',count);
            
end