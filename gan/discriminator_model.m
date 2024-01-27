% Discriminator Architecture
function dlYDisc = modelDiscriminator(dlX,parameters)
    dlYDisc = dlconv(dlX,parameters.Conv1.Weights,parameters.Conv1.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    dlYDisc = dlconv(dlYDisc,parameters.Conv2.Weights,parameters.Conv2.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    dlYDisc = dlconv(dlYDisc,parameters.Conv3.Weights,parameters.Conv3.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    dlYDisc = dlconv(dlYDisc,parameters.Conv4.Weights,parameters.Conv4.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    dlYDisc = dlconv(dlYDisc,parameters.Conv5.Weights,parameters.Conv5.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
     
    dlYDisc = stripdims(dlYDisc);
    dlYDisc = permute(dlYDisc,[3 2 1 4]);
    dlYDisc = reshape(dlYDisc,4*4*64*16,numel(dlYDisc)/(4*4*64*16));
    
    weights = parameters.FC.Weights;
    bias = parameters.FC.Bias;
    dlYDisc = fullyconnect(dlYDisc,weights,bias,Dataformat="CB");
end


% Discriminator Gradients
function gradientsDiscriminator = modelDiscriminatorGradients(discriminatorParameters,generatorParameters,X,Z)
    % Calculate the predictions for real data with the discriminator network.
    Y = modelDiscriminator(X,discriminatorParameters);
    
    % Calculate the predictions for generated data with the discriminator network.
    Xgen = modelGenerator(Z,generatorParameters);
    Ygen = modelDiscriminator(dlarray(Xgen,"SSCB"),discriminatorParameters);
    
    % Calculate the GAN loss.
    lossDiscriminator = ganDiscriminatorLoss(Y,Ygen);
    
    % For each network, calculate the gradients with respect to the loss.
    gradientsDiscriminator = dlgradient(lossDiscriminator,discriminatorParameters);
end


% Discriminator Loss Function
function  lossDiscriminator = ganDiscriminatorLoss(dlYPred,dlYPredGenerated)
    fake = dlarray(zeros(1,size(dlYPred,2)));
    real = dlarray(ones(1,size(dlYPred,2)));
    
    D_loss = mean(sigmoid_cross_entropy_with_logits(dlYPredGenerated,fake));
    D_loss = D_loss + mean(sigmoid_cross_entropy_with_logits(dlYPred,real));
    lossDiscriminator  = D_loss / 2;
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