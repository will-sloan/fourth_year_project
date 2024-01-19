
% Current expectation: (128, 128) image

miniBatchSize = 64;
inputSize = [128 512];
numChannels = 1;
X = rand(inputSize(1),inputSize(2),numChannels,miniBatchSize);
X = dlarray(X,"SSCB");

% [128, 512, 1, 64]

discParams = initializeDiscriminatorWeights;

weights = discParams.Conv1.Weights;
bias = discParams.Conv1.Bias;

Y = modelDiscriminator(X, discParams);
output = size(Y)


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



% ================ Discriminator Code ================
% Discriminator Architecture
function dlYDisc = modelDiscriminator(dlX,parameters)

    size_dlX_init = size(dlX)

    dlYDisc = dlconv(dlX,parameters.Conv1.Weights,parameters.Conv1.Bias,Stride=2,Padding="same");
    % dlYDisc = dlconv(dlX,parameters.Conv1.Weights,parameters.Conv1.Bias,Stride=[2, 8], Padding=[2, 2]);
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    size_dlYDisc_1 = size(dlYDisc)

    dlYDisc = dlconv(dlYDisc,parameters.Conv2.Weights,parameters.Conv2.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    size_dlYDisc_2 = size(dlYDisc)

    dlYDisc = dlconv(dlYDisc,parameters.Conv3.Weights,parameters.Conv3.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    size_dlYDisc_3 = size(dlYDisc)

    dlYDisc = dlconv(dlYDisc,parameters.Conv4.Weights,parameters.Conv4.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    size_dlYDisc_4 = size(dlYDisc)

    dlYDisc = dlconv(dlYDisc,parameters.Conv5.Weights,parameters.Conv5.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
     
    size_dlYDisc_5 = size(dlYDisc)

    dlYDisc = stripdims(dlYDisc);

    size_dlYDiscStripped = size(dlYDisc)

    dlYDisc = permute(dlYDisc,[3 2 1 4]);

    size_dlYDiscPermuted = size(dlYDisc)

    dlYDisc = reshape(dlYDisc,(4*4*16*2)*(64*2),numel(dlYDisc)/((4*4*16*2)*(64*2)));
    
    size_discReshape = size(dlYDisc)

    weights = parameters.FC.Weights;
    bias = parameters.FC.Bias;
    dlYDisc = fullyconnect(dlYDisc,weights,bias,Dataformat="CB");
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
    weights = iGlorotInitialize([1, (8 * 16) * (dim * 8)]);
    bias = zeros(1,1,"single");
    discriminatorParameters.FC.Weights = dlarray(weights);
    discriminatorParameters.FC.Bias = dlarray(bias);
end



%{
% Discriminator Architecture
function dlYDisc = modelDiscriminator(dlX,parameters)

    size_dlX_init = size(dlX)

    dlYDisc = dlconv(dlX,parameters.Conv1.Weights,parameters.Conv1.Bias,Stride=2,Padding="same");
    % dlYDisc = dlconv(dlX,parameters.Conv1.Weights,parameters.Conv1.Bias,Stride=[2, 8], Padding=[2, 2]);
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    size_dlYDisc_1 = size(dlYDisc)

    dlYDisc = dlconv(dlYDisc,parameters.Conv2.Weights,parameters.Conv2.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    size_dlYDisc_2 = size(dlYDisc)

    dlYDisc = dlconv(dlYDisc,parameters.Conv3.Weights,parameters.Conv3.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    size_dlYDisc_3 = size(dlYDisc)

    dlYDisc = dlconv(dlYDisc,parameters.Conv4.Weights,parameters.Conv4.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    size_dlYDisc_4 = size(dlYDisc)

    dlYDisc = dlconv(dlYDisc,parameters.Conv5.Weights,parameters.Conv5.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
     
    size_dlYDisc_5 = size(dlYDisc)

    dlYDisc = stripdims(dlYDisc);

    size_dlYDiscStripped = size(dlYDisc)

    dlYDisc = permute(dlYDisc,[3 2 1 4]);

    size_dlYDiscPermuted = size(dlYDisc)

    dlYDisc = reshape(dlYDisc,4*4*64*16,numel(dlYDisc)/(4*4*64*16));
    
    size_discReshape = size(dlYDisc)

    weights = parameters.FC.Weights;
    bias = parameters.FC.Bias;
    dlYDisc = fullyconnect(dlYDisc,weights,bias,Dataformat="CB");
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
%}