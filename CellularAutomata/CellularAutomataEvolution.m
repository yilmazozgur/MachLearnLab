%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
%This function computes cellular automata state evolution
%
%%% Input:
%RawFeatures : A struct of the features directlt from data or after some
%   feature computation.
%DataStats:statistics of data for normalization
%options: struct for options. 
%options.SparsifyFlag: flag for sparsification of neural responses. 
%   Default=true.
%options.SparsityMultiplier: the amounf of sparsity. Default=0.7.
%options.GPU: enable GPU computing. Default=false.
%options.BatchSize: the number of images to be processed together. Default=20;
%options.TrainOrTest: Train or Test data will be worked on. Default: 'Train'.
%options.SaveFeatures: Save the results to the Root when finished.
%   Default=false.
%options.SavePath: the path to save the results if SaveRFs flag is up.
%   Default='';
%
%%% Output:
%NeuralNetFeatures: A struct that holds neural network responses of size
%   ImageDim1*ImageDim1*numRFs for each image.
%
%
%From:
%TOU_ML
%Ozgur Yilmaz, Turgut Ozal University, Ankara
%Web: ozguryilmazresearch.net
%Ref: It is based on Adam Coates code, Coates and Ng, 2011 paper.
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function CAStates = CellularAutomataEvolution(InitialStates, CAStates, RuleArray, RandomPermutation, SamplingMask, options)

%compute important info on data and CA evolution
NoOfDistinctInputs = size(InitialStates,1);
LengthFeature = size(InitialStates,2);
NoOfPermutations=size(RandomPermutation,1);

%Initialize 
%used for evolution
siz_vec=[2 2 2];
k = [1 cumprod(siz_vec(1:end-1))];
%Single step CA evolution vector
CAStates_Iter=CAStates(:,1:NoOfPermutations*LengthFeature);
% %Whole state matrix
% CAStates = zeros(NoOfDistinctInputs, LengthFeature*NoOfPermutations*options.CAIter);

%change type for faster in mex'ed version
% input_vec=int8(input_vec);
% CAStates_Iter=int8(CAStates_Iter);
% RuleArray=int8(RuleArray);
% CAStates=int8(CAStates);

%put every random permutation one after another, create a big redundant
%feature vector
%add 1 for evolution
InitialStates=InitialStates+1;
for pe=1:1:NoOfPermutations
    RandPermutTemp=RandomPermutation(pe,1:LengthFeature);
    InitialStates_=InitialStates(:,RandPermutTemp);
    CAStates_Iter(:,(pe-1)*LengthFeature+1:pe*LengthFeature)=InitialStates_; %add 1 for Ele. CA processing
end

% Loop over options.CAIter generations.
%evolve CA states, record all states in CAStates
for t = 1:options.CAIter
    CAStates(:,(t-1)*LengthFeature*NoOfPermutations+1:t*LengthFeature*NoOfPermutations)=CAStates_Iter;
    ind = 1;
    Abc=[CAStates_Iter(:,2:end) CAStates_Iter(:,1)]-1;
    ind = ind + (Abc)*k(1);
    ind = ind + (CAStates_Iter-1)*k(2);
    Abc2=[CAStates_Iter(:,end) CAStates_Iter(:,1:end-1)]-1;
    ind = ind + (Abc2)*k(3);
    CAStates_Iter=RuleArray(ind);
%     CAStates(:,(t-1)*LengthFeature*NoOfPermutations+1:t*LengthFeature*NoOfPermutations)=CAStates_Iter;
end

%back to ones and zeros
CAStates=CAStates-1;

if not(isempty(SamplingMask)) %not sampled if sampling mask is not given
    %sample the state space for reduction
    CAStates=CAStates(:,SamplingMask);
end


