
trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];

dlZ = dlarray(2 * ( rand(1,1,100,1,"single") - 0.5 ));

gen_params = initializeGeneratorWeights();

modelGenerator(dlZ, gen_params)

% 1 Generate a dlarray object containing an array of random values for the generator network.

% 2 For GPU training, convert the data to a gpuArray (Parallel Computing Toolbox) object.

% 3 Evaluate the model gradients using dlfeval (Deep Learning Toolbox) and the helper functions, modelDiscriminatorGradients and modelGeneratorGradients.

% 4 Update the network parameters using the adamupdate (Deep Learning Toolbox) function.


%% MATLAB GAN Training Method

iteration = 0;

doTraining = true;
saveCheckpoints = true;

numLatentInputs = 100;

for epoch = 1:maxEpochs

    % Shuffle the data.
    idx = randperm(size(STrain,4));
    STrain = STrain(:,:,:,idx);

    % Loop over mini-batches.
    for index = 1:numIterationsPerEpoch
        
        iteration = iteration + 1;

        % Read mini-batch of data.
        dlX = STrain(:,:,:,(index-1)*miniBatchSize+1:index*miniBatchSize);
        dlX = dlarray(dlX,"SSCB");
        
        % Generate latent inputs for the generator network.
        Z = 2 * ( rand(1,1,numLatentInputs,miniBatchSize,"single") - 0.5 ) ;
        dlZ = dlarray(Z);

        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlZ = gpuArray(dlZ);
            dlX = gpuArray(dlX);
        end
        
        % Evaluate the discriminator gradients using dlfeval and the
        % modelDiscriminatorGradients helper function.
        gradientsDiscriminator = ...
            dlfeval(@modelDiscriminatorGradients,discriminatorParameters,generatorParameters,dlX,dlZ);
        
        % Update the discriminator network parameters.
        [discriminatorParameters,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(discriminatorParameters,gradientsDiscriminator, ...
            trailingAvgDiscriminator,trailingAvgSqDiscriminator,iteration, ...
            learnRateDiscriminator,gradientDecayFactor,squaredGradientDecayFactor);

        % Generate latent inputs for the generator network.
        Z = 2 * ( rand(1,1,numLatentInputs,miniBatchSize,"single") - 0.5 ) ;
        dlZ = dlarray(Z);
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlZ = gpuArray(dlZ);
        end
        
        % Evaluate the generator gradients using dlfeval and the
        % |modelGeneratorGradients| helper function.
        gradientsGenerator  = ...
            dlfeval(@modelGeneratorGradients,discriminatorParameters,generatorParameters,dlZ);
        
        % Update the generator network parameters.
        [generatorParameters,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(generatorParameters,gradientsGenerator, ...
            trailingAvgGenerator,trailingAvgSqGenerator,iteration, ...
            learnRateGenerator,gradientDecayFactor,squaredGradientDecayFactor);
    end

    % Every 10 epochs, save a training snapshot to a MAT file.
    if mod(epoch,10)==0
        disp("Epoch " + epoch + " out of " + maxEpochs + " complete.");
        if saveCheckpoints
            % Save checkpoint in case training is interrupted.
            save("audiogancheckpoint.mat", ...
                "generatorParameters","discriminatorParameters", ...
                "trailingAvgDiscriminator","trailingAvgSqDiscriminator", ...
                "trailingAvgGenerator","trailingAvgSqGenerator","iteration");
        end
    end
end