function NeuralNetFeatures=ExtractBRIEF(Dataset,ReceptiveFields,DataStats,options)
Type=options{1,9}.Type; % Type of BRIEF Feature Extraction
 switch Type
     case 'BriefSubImages'
         NeuralNetFeatures=BriefSubImages(Dataset,ReceptiveFields,DataStats,options);
     case 'BriefOverlappedPatches'
         NeuralNetFeatures=BriefOverlappedPatches(Dataset,ReceptiveFields,DataStats,options);
 end 
end%end function