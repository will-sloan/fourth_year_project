
% Generator Architecture
function dlYGen = modelGenerator(dlX,parameters)

    dlYGen = fullyconnect(dlX,parameters.FC.Weights,parameters.FC.Bias,Dataformat="SSCB");
    
    dlYGen = reshape(dlYGen,[1024 4 4 size(dlYGen,2)]);
    dlYGen = permute(dlYGen,[3 2 1 4]);
    dlYGen = relu(dlYGen);
    
    dlYGen = dltranspconv(dlYGen,parameters.Conv1.Weights,parameters.Conv1.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = relu(dlYGen);
    
    dlYGen = dltranspconv(dlYGen,parameters.Conv2.Weights,parameters.Conv2.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = relu(dlYGen);
    
    dlYGen = dltranspconv(dlYGen,parameters.Conv3.Weights,parameters.Conv3.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = relu(dlYGen);
    
    dlYGen = dltranspconv(dlYGen,parameters.Conv4.Weights,parameters.Conv4.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = relu(dlYGen);
    
    dlYGen = dltranspconv(dlYGen,parameters.Conv5.Weights,parameters.Conv5.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    dlYGen = tanh(dlYGen);
end

% Generator Gradients
function gradientsGenerator = modelGeneratorGradients(discriminatorParameters,generatorParameters,Z)
    % Calculate the predictions for generated data with the discriminator network.
    Xgen = modelGenerator(Z,generatorParameters);
    Ygen = modelDiscriminator(dlarray(Xgen,"SSCB"),discriminatorParameters);
    
    % Calculate the GAN loss
    lossGenerator = ganGeneratorLoss(Ygen);
    
    % For each network, calculate the gradients with respect to the loss.
    gradientsGenerator = dlgradient(lossGenerator,generatorParameters);
end

% Generator Loss
function lossGenerator = ganGeneratorLoss(dlYPredGenerated)
    real = dlarray(ones(1,size(dlYPredGenerated,2)));
    lossGenerator = mean(sigmoid_cross_entropy_with_logits(dlYPredGenerated,real));
end

% Generator Weights Initializer
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