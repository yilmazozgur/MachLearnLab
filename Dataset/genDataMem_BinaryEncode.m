function dataset = genDataMem_BinaryEncode(T, D, N, S)
% generating training data for the Schmidhuber addition task and variants
%
% T is length of distractor sequence
% D is nr of to-be-memorized input symbols
% N is length of to-be-memorized period
% S is Nr of generated samples
% dataset is array of size (T+2*N) x (2*(D+2)) x S, where the first D+2
% columns per page contain the input and the last D+2 the output.
% The first D channels in input/output code the "payload" input, the
% channel D+1 the distractor input, and the channel D+2 the cue.
SizeBinary=ceil(log2(D));
L = T + 2*N;
dataset = zeros(L, 2*(SizeBinary+2), S);
for i = 1:S
    %%% input
    % compute binary payload sequence
    bits = randi([1,D],N);
    % set first N inputs
    
    for t = 1:N
        StrBin=dec2bin(bits(t),SizeBinary);
        for ii=1:1:SizeBinary
            dataset(t,ii,i) = str2double(StrBin(ii));
        end
    end
    % set dummy input
    dataset(N+1:end,SizeBinary+1,i) = 1;
    % set trigger input
    dataset(end-N,SizeBinary+1,i) = 0;
    dataset(end-N,SizeBinary+2,i) = 1;
    
    %%% output
    dataset(1:end-N,end-1,i) = 1;
    for t = 1:N
        StrBin=dec2bin(bits(t),SizeBinary);
        for ii=1:1:SizeBinary
            dataset(end-N+t, (SizeBinary+2)+ii,i) = str2double(StrBin(ii));
        end
    end
    
end