function [x1,x2,y1,y2,z1,z2]=distribution3(subn,Number_Of_Tests,FileName,Dimension)
fprintf('Generating a new distribution!!!')
% subn = 10;

S=subn/2;
test=max(Number_Of_Tests);
% Dimension='3D';
    if strcmp(Dimension,'3D')
                len=subn*subn*3;
                v1=1:len;
                A=nchoosek(v1,2);
                cn=0;
                for i=floor(-S+1):floor(S)
                    for j=floor(-S+1):floor(S)
                        for k=1:3
                            cn=cn+1;
                            B(cn,:)=[i j k];
                        end
                    end
                end

                f=A(randperm(length(A),test),1);
                g=A(randperm(length(A),test),2);

                M=[B(f,:) B(g,:)];

                x1=M(:,1);
                x2=M(:,4);
                y1=M(:,2);
                y2=M(:,5);
                z1=M(:,3);
                z2=M(:,6);
                save(FileName,'x1','x2','y1','y2','z1','z2');
         
         
    else %Dimension='2D'
        len=subn*subn;
        v1=1:len;
        A=nchoosek(v1,2);
        cn=0;
        for i=floor(-S+1):floor(S)
            for j=floor(-S+1):floor(S)
                    cn=cn+1;
                    B(cn,:)=[i j];
            end
        end
        f=A(randperm(length(A),test),1);
        g=A(randperm(length(A),test),2);

        M=[B(f,:) B(g,:)];
        
        x1=M(:,1);
        x2=M(:,3);
        y1=M(:,2);
        y2=M(:,4);
        save(FileName,'x1','x2','y1','y2');
    end

    
   




end
% B(f)

% C=B(randperm(cn,test),:);