%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Explanation:
% This function makes a decision about which extract type we are going to
% use in order to ExtractFeatures:
% 'ExtractBRIEF'
% 'ExtractORB'
% 'ExtractPoolNormalizeSingleLayerFeature_v3'
%  etc.
%
%From:
%TOU_ML
%Ozgur Yilmaz, Turgut Ozal University, Ankara
%Web: ozguryilmazresearch.net
%Ref: It is based on Adam Coates code, Coates and Ng, 2011 paper.
%May 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function NeuralNetFeatures=LabExtractFeatures(Dataset,ReceptiveFields,DataStats,options)
Type=options{1,2}.Type;
if (options{8}.BinaryDescription==1) %If Binary Description of RFs options is true
    NeuralNetFeatures = ExtractPoolNormalizeBinaryFeatures(Dataset, ReceptiveFields, DataStats, options);
else %If Binary Description of RFs options is false
     switch Type
         case 'ExtractPoolNormalizeSingleLayerFeature_v3'
             NeuralNetFeatures = ExtractPoolNormalizeSingleLayerFeature_v3(Dataset,ReceptiveFields,DataStats,options);
         case 'ExtractBRIEF'
             NeuralNetFeatures = ExtractBRIEF(Dataset,ReceptiveFields,DataStats,options);
     end
end
 
end%end function


