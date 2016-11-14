function PooledFeature=HalveHalveAndPool(FeatureImage)

% find rows and cols
prows = size(FeatureImage,2);
pcols = size(FeatureImage,3);

% compute half sizes
halfr = round(prows/2);
halfc = round(pcols/2);

%halfhalf sizes
halfrhalf=round(halfr/2);
halfchalf=round(halfc/2);

q1_q1 = sum(sum(FeatureImage(:,1:halfrhalf, 1:halfchalf,:), 2),3);
q1_q2 = sum(sum(FeatureImage(:,halfrhalf+1:halfr, 1:halfchalf, :), 2),3);
q1_q3 = sum(sum(FeatureImage(:,1:halfrhalf, halfchalf+1:halfc, :), 2),3);
q1_q4 = sum(sum(FeatureImage(:,halfrhalf+1:halfr, halfchalf+1:halfc, :), 2),3);
q2_q1 = sum(sum(FeatureImage(:,halfr+1:halfr+halfrhalf, 1:halfchalf, :), 2),3);
q2_q2 = sum(sum(FeatureImage(:,halfr+halfrhalf+1:end, 1:halfchalf, :), 2),3);
q2_q3 = sum(sum(FeatureImage(:,halfr+1:halfr+halfrhalf, halfchalf+1:halfc, :), 2),3);
q2_q4 = sum(sum(FeatureImage(:,halfr+halfrhalf+1:end, halfchalf+1:halfc, :), 2),3);
q3_q1 = sum(sum(FeatureImage(:,1:halfrhalf, halfc+1:halfc+halfchalf, :), 2),3);
q3_q2 = sum(sum(FeatureImage(:,halfrhalf+1:halfr, halfc+1:halfc+halfchalf, :), 2),3);
q3_q3 = sum(sum(FeatureImage(:,1:halfrhalf, halfc+halfchalf+1:end, :), 2),3);
q3_q4 = sum(sum(FeatureImage(:,halfrhalf+1:halfr, halfc+halfchalf+1:end, :), 2),3);
q4_q1 = sum(sum(FeatureImage(:,halfr+1:halfr+halfrhalf, halfc+1:halfc+halfchalf, :), 2),3);
q4_q2 = sum(sum(FeatureImage(:,halfr+halfrhalf+1:end, halfc+1:halfc+halfchalf, :), 2),3);
q4_q3 = sum(sum(FeatureImage(:,halfr+1:halfr+halfrhalf, halfc+halfchalf+1:end, :), 2),3);
q4_q4 = sum(sum(FeatureImage(:,halfr+halfrhalf+1:end, halfc+halfchalf+1:end, :), 2),3);
PooledFeature = [squeeze(q1_q1) squeeze(q1_q2) squeeze(q1_q3) squeeze(q1_q4) squeeze(q2_q1) squeeze(q2_q2) squeeze(q2_q3) squeeze(q2_q4)...
    squeeze(q3_q1) squeeze(q3_q2) squeeze(q3_q3) squeeze(q3_q4) squeeze(q4_q1) squeeze(q4_q2) squeeze(q4_q3) squeeze(q4_q4)];


