

miniBatchSize = 64;
inputSize = [199 320];
numChannels = 1;
X = rand(inputSize(1),inputSize(2),numChannels,miniBatchSize);
X = dlarray(X,"SSCB");


filterSize = [10 9];
numFilters = 64;
weights = rand(filterSize(1),filterSize(2),numChannels,numFilters);
bias = zeros(1,numFilters);

Y = dlconv(X, weights, bias, Stride=[3, 5], Padding=[1, 2]);
size(Y)


function w = iGlorotInitialize(sz)
if numel(sz) == 2 % fully-connected
    numInputs = sz(2);
    numOutputs = sz(1);
else % Convolutional Layers
    numInputs = prod(sz(1:3));
    numOutputs = prod(sz([1 2 4]));
end
multiplier = sqrt(2 / (numInputs + numOutputs));
w = multiplier * sqrt(3) * (2 * rand(sz,"single") - 1);
end


% Discriminator Weights Initializer
function discriminatorParameters = initializeDiscriminatorWeights
    filterSize = [5 5];
    dim = 64;
    
    % Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) 1 dim]);
    bias = zeros(1,1,dim,"single");
    discriminatorParameters.Conv1.Weights = dlarray(weights);
    discriminatorParameters.Conv1.Bias = dlarray(bias);
    
    % Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) dim 2*dim]);
    bias = zeros(1,1,2*dim,"single");
    discriminatorParameters.Conv2.Weights = dlarray(weights);
    discriminatorParameters.Conv2.Bias = dlarray(bias);
    
    % Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) 2*dim 4*dim]);
    bias = zeros(1,1,4*dim,"single");
    discriminatorParameters.Conv3.Weights = dlarray(weights);
    discriminatorParameters.Conv3.Bias = dlarray(bias);
    
    % Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) 4*dim 8*dim]);
    bias = zeros(1,1,8*dim,"single");
    discriminatorParameters.Conv4.Weights = dlarray(weights);
    discriminatorParameters.Conv4.Bias = dlarray(bias);
    
    % Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) 8*dim 16*dim]);
    bias = zeros(1,1,16*dim,"single");
    discriminatorParameters.Conv5.Weights = dlarray(weights);
    discriminatorParameters.Conv5.Bias = dlarray(bias);
    
    % fully connected
    weights = iGlorotInitialize([1,4 * 4 * dim * 16]);
    bias = zeros(1,1,"single");
    discriminatorParameters.FC.Weights = dlarray(weights);
    discriminatorParameters.FC.Bias = dlarray(bias);
end