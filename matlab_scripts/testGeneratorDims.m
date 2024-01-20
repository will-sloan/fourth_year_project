miniBatchSize = 64;
numChannels = 1;
numLatentInputs = 100;

% Generate latent inputs for the generator network.
Z = 2 * ( rand(1,1,numLatentInputs,miniBatchSize,"single") - 0.5 ) ;
dlZ = dlarray(Z);

% [128, 512, 1, 64]

genParams = initializeGeneratorWeights;

weights = genParams.Conv1.Weights;
bias = genParams.Conv1.Bias;

sizeaz = size(weights)

Ygen = modelGenerator(dlZ, genParams);
output = size(Ygen)


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


% ================ Generator Code ================
% Generator Architecture
function dlYGen = modelGenerator(dlX,parameters)

    size_dlZ_init = size(dlX)

    dlYGen = fullyconnect(dlX,parameters.FC.Weights,parameters.FC.Bias,Dataformat="SSCB");
    
    size_Gen_FC = size(dlYGen)

    dlYGen = reshape(dlYGen,[1024 16 4 size(dlYGen,2)]);

    size_Gen_Reshape = size(dlYGen)

    dlYGen = permute(dlYGen,[3 2 1 4]);

    size_Gen_permuted = size(dlYGen)

    dlYGen = relu(dlYGen);
    
    dlYGen = dltranspconv(dlYGen,parameters.Conv1.Weights,parameters.Conv1.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = relu(dlYGen);
    
    size_Gen_TC1 = size(dlYGen)

    dlYGen = dltranspconv(dlYGen,parameters.Conv2.Weights,parameters.Conv2.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = relu(dlYGen);
    
    size_Gen_TC2 = size(dlYGen)

    dlYGen = dltranspconv(dlYGen,parameters.Conv3.Weights,parameters.Conv3.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = relu(dlYGen);
    
    size_Gen_TC3 = size(dlYGen)

    dlYGen = dltranspconv(dlYGen,parameters.Conv4.Weights,parameters.Conv4.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = relu(dlYGen);
    
    size_Gen_TC4 = size(dlYGen)

    dlYGen = dltranspconv(dlYGen,parameters.Conv5.Weights,parameters.Conv5.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = tanh(dlYGen);
end

% Generator Weights Initializer
function generatorParameters = initializeGeneratorWeights
    dim = 64;
    
    % Dense 1
    weights = iGlorotInitialize([dim*2*512,100]);
    bias = zeros(dim*2*512,1,"single");
    generatorParameters.FC.Weights = dlarray(weights);
    generatorParameters.FC.Bias = dlarray(bias);
    
    filterSize = [5 5];
    
    % Trans Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) 8*dim 16*dim]);
    bias = zeros(1,1,dim*8,"single");
    generatorParameters.Conv1.Weights = dlarray(weights);
    generatorParameters.Conv1.Bias = dlarray(bias);
    
    % Trans Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) 4*dim 8*dim]);
    bias = zeros(1,1,dim*4,"single");
    generatorParameters.Conv2.Weights = dlarray(weights);
    generatorParameters.Conv2.Bias = dlarray(bias);
    
    % Trans Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) 2*dim 4*dim]);
    bias = zeros(1,1,dim*2,"single");
    generatorParameters.Conv3.Weights = dlarray(weights);
    generatorParameters.Conv3.Bias = dlarray(bias);
    
    % Trans Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) dim 2*dim]);
    bias = zeros(1,1,dim,"single");
    generatorParameters.Conv4.Weights = dlarray(weights);
    generatorParameters.Conv4.Bias = dlarray(bias);
    
    % Trans Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) 1 dim]);
    bias = zeros(1,1,1,"single");
    generatorParameters.Conv5.Weights = dlarray(weights);
    generatorParameters.Conv5.Bias = dlarray(bias);
end


%{

% Generator Architecture
function dlYGen = modelGenerator(dlX,parameters)

    size_dlZ_init = size(dlX)

    dlYGen = fullyconnect(dlX,parameters.FC.Weights,parameters.FC.Bias,Dataformat="SSCB");
    
    size_Gen_FC = size(dlYGen)

    dlYGen = reshape(dlYGen,[1024 4 4 size(dlYGen,2)]);

    size_Gen_Reshape = size(dlYGen)

    dlYGen = permute(dlYGen,[3 2 1 4]);

    size_Gen_permuted = size(dlYGen)

    dlYGen = relu(dlYGen);
    
    dlYGen = dltranspconv(dlYGen,parameters.Conv1.Weights,parameters.Conv1.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = relu(dlYGen);
    
    size_Gen_TC1 = size(dlYGen)

    dlYGen = dltranspconv(dlYGen,parameters.Conv2.Weights,parameters.Conv2.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = relu(dlYGen);
    
    size_Gen_TC2 = size(dlYGen)

    dlYGen = dltranspconv(dlYGen,parameters.Conv3.Weights,parameters.Conv3.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = relu(dlYGen);
    
    size_Gen_TC3 = size(dlYGen)

    dlYGen = dltranspconv(dlYGen,parameters.Conv4.Weights,parameters.Conv4.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = relu(dlYGen);
    
    size_Gen_TC4 = size(dlYGen)

    dlYGen = dltranspconv(dlYGen,parameters.Conv5.Weights,parameters.Conv5.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = tanh(dlYGen);
end




function generatorParameters = initializeGeneratorWeights
    dim = 64;
    
    % Dense 1
    weights = iGlorotInitialize([dim*256,100]);
    bias = zeros(dim*256,1,"single");
    generatorParameters.FC.Weights = dlarray(weights);
    generatorParameters.FC.Bias = dlarray(bias);
    
    filterSize = [5 5];
    
    % Trans Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) 8*dim 16*dim]);
    bias = zeros(1,1,dim*8,"single");
    generatorParameters.Conv1.Weights = dlarray(weights);
    generatorParameters.Conv1.Bias = dlarray(bias);
    
    % Trans Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) 4*dim 8*dim]);
    bias = zeros(1,1,dim*4,"single");
    generatorParameters.Conv2.Weights = dlarray(weights);
    generatorParameters.Conv2.Bias = dlarray(bias);
    
    % Trans Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) 2*dim 4*dim]);
    bias = zeros(1,1,dim*2,"single");
    generatorParameters.Conv3.Weights = dlarray(weights);
    generatorParameters.Conv3.Bias = dlarray(bias);
    
    % Trans Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) dim 2*dim]);
    bias = zeros(1,1,dim,"single");
    generatorParameters.Conv4.Weights = dlarray(weights);
    generatorParameters.Conv4.Bias = dlarray(bias);
    
    % Trans Conv2D
    weights = iGlorotInitialize([filterSize(1) filterSize(2) 1 dim]);
    bias = zeros(1,1,1,"single");
    generatorParameters.Conv5.Weights = dlarray(weights);
    generatorParameters.Conv5.Bias = dlarray(bias);
end
%}