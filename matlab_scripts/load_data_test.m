


data = load('audiogancheckpoint9.mat');

discVals = data.epochDiscriminatorSample;

% 1, 2, 180
realVals = permute(discVals, [1, 3, 2]);

size(realVals(1, :, 1))

Y = realVals(1, :, 1);
YGen = realVals(1, :, 2);

L = 1:numel(Y);

size(L)

YNorm = 1./(1+exp(-Y));

YGenNorm = 1./(1+exp(-YGen));

plot(L, YNorm(L), L, YGenNorm(L));

xlabel("Epoch");
ylabel("Discriminator Output");
legend("Real Data","Generated Data");





% plot(X, y);