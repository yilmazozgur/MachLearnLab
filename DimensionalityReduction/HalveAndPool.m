function PooledFeature=HalveAndPool(FeatureImage)

% find rows and cols
prows = size(FeatureImage,2);
pcols = size(FeatureImage,3);

% compute half sizes
halfr = round(prows/2);
halfc = round(pcols/2);

q1 = sum(sum(FeatureImage(:,1:halfr, 1:halfc, :), 2),3);
q2 = sum(sum(FeatureImage(:,halfr+1:end, 1:halfc, :), 2),3);
q3 = sum(sum(FeatureImage(:,1:halfr, halfc+1:end, :), 2),3);
q4 = sum(sum(FeatureImage(:,halfr+1:end, halfc+1:end, :), 2),3);
PooledFeature = [squeeze(q1) squeeze(q2) squeeze(q3) squeeze(q4)];


