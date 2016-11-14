function dataset = genDataMem(T, D, N, S)
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

L = T + 2*N;
dataset = zeros(L, 2*(D+2), S);
for i = 1:S
    %%% input
    % compute binary payload sequence
    bits = randi([1,D],N);
    % set first N inputs
    for t = 1:N
        dataset(t,bits(t),i) = 1;
    end
    % set dummy input
    dataset(N+1:end,D+1,i) = 1;
    % set trigger input
    dataset(end-N,D+1,i) = 0;
    dataset(end-N,D+2,i) = 1;
    
    %%% output
    dataset(1:end-N,end-1,i) = 1;
    for t = 1:N
        dataset(end-N+t, (D+2)+bits(t),i) = 1;
    end
    
end