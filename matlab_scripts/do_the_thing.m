test = the_thing

function y = the_thing
    persistent pGeneratorParameters pMean pSTD
        if isempty(pGeneratorParameters)
            % If the MAT file does not exist, download it
            filename = "drumGeneratorWeights.mat";
            load(filename,"SMean","SStd","generatorParameters");
            pMean = SMean;
            pSTD  = SStd;
            pGeneratorParameters = generatorParameters;
        end
end

function uhh

    % 0 Load Training Data
    path = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_1\\%dDeg_EARS_1_%d.wav", angle, angle, chunk_no)
    path2 = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_2\\%dDeg_EARS_2_%d.wav", angle, angle, chunk_no);
    
    if angle == 0 || angle == 15
        path = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_1\\%dDeg_EARSFullAudioRecording_1_%d.wav", angle, angle, chunk_no)
        path2 = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_2\\%dDeg_EARSFullAudioRecording_2_%d.wav", angle, angle, chunk_no);
    end
    

    % Change this to match our input data
    ads = audioDatastore(percussivesoundsFolder,IncludeSubfolders=true);

    % 1 Define the STFT parameters.
    fftLength = 256;
    win = hann(fftLength,"periodic");
    overlapLength = 128;
    
    % 2 First, determine the number of partitions for the dataset. If 
    % you do not have Parallel Computing Toolbox™, use a single partition.
    if canUseParallelPool
        pool = gcp;
        numPar = numpartitions(ads,pool);
    else
        numPar = 1;
    end


    % 3 For each partition, read from the datastore and compute the STFT.
    parfor ii = 1:numPar
        subds = partition(ads,numPar,ii);
        STrain = zeros(fftLength/2+1,128,1,numel(subds.Files));
        
        for idx = 1:numel(subds.Files)
            
            % Read audio
            [x,xinfo] = read(subds);
    
            % Preprocess
            x = preprocessAudio(single(x),xinfo.SampleRate);
    
            % STFT
            S0 = stft(x,Window=win,OverlapLength=overlapLength,FrequencyRange="onesided");
            
            % Magnitude
            S = abs(S0);
    
            STrain(:,:,:,idx) = S; % Populates the 4th dimension with stft data
        end
        STrainC{ii} = STrain; % Contains all the 4D arrays with 3 dimensions of 0s and 1 dimension of stft data
    end

    % 4 Convert the output to a four-dimensional array with STFTs along the fourth dimension.
    STrain = cat(4,STrainC{:});

    % 5 Convert the data to the log scale to better align with human perception.
    STrain = log(STrain + 1e-6);

    % 6 Compute the STFT mean and standard deviation of each frequency bin.
    SMean = mean(STrain,[2 3 4]);
    SStd = std(STrain,1,[2 3 4]);

    % 7 Normalize each frequency bin.
    STrain = (STrain-SMean)./SStd;

    % 8 The computed STFTs have unbounded values. Following the approach 
    % in [1], make the data bounded by clipping the spectra to 3 standard 
    % deviations and rescaling to [-1 1].
    STrain = STrain/3;
    Y = reshape(STrain,numel(STrain),1);
    Y(Y<-1) = -1;
    Y(Y>1) = 1;
    STrain = reshape(Y,size(STrain));

    % 9 Discard the last frequency bin to force the number of STFT bins 
    % to a power of two (which works well with convolutional layers).
    STrain = STrain(1:end-1,:,:,:);

    % 10 Permute the dimensions in preparation for feeding to the 
    % discriminator.
    STrain = permute(STrain,[2 1 3 4]);

end


% ========== Specify Training Options ==========
function training_options

    % 1 Train with a mini-batch size of 64 for 1000 epochs.
    maxEpochs = 1000;
    miniBatchSize = 64;


    % 2 Compute the number of iterations required to consume the data.
    numIterationsPerEpoch = floor(size(STrain,4)/miniBatchSize);


    % 3 Specify the options for Adam optimization. Set the learn rate of 
    % the generator and discriminator to 0.0002. For both networks, use a 
    % gradient decay factor of 0.5 and a squared gradient decay factor of 0.999.
    learnRateGenerator = 0.0002;
    learnRateDiscriminator = 0.0002;
    gradientDecayFactor = 0.5;
    squaredGradientDecayFactor = 0.999;


    % 4 Train on a GPU if one is available. Using a GPU requires 
    % Parallel Computing Toolbox™.
    executionEnvironment = auto;


    % 5 Initialize the generator and discriminator weights. The 
    % initializeGeneratorWeights and initializeDiscriminatorWeights 
    % functions return random weights obtained using Glorot uniform 
    % initialization. The functions are included at the end of this example.
    generatorParameters = initializeGeneratorWeights;
    discriminatorParameters = initializeDiscriminatorWeights;
end


function train_model

    % 1 Initialize the parameters for Adam.
    trailingAvgGenerator = [];
    trailingAvgSqGenerator = [];
    trailingAvgDiscriminator = [];
    trailingAvgSqDiscriminator = [];


    % 2 Depending on your machine, training this network can take hours. 
    % To skip training, set doTraining to false.
    doTraining = true;


    % 3 You can set saveCheckpoints to true to save the updated weights and 
    % states to a MAT file every ten epochs. You can then use this MAT file 
    % to resume training if it is interrupted.
    saveCheckpoints = true;


    % 4 You can set saveCheckpoints to true to save the updated weights and
    % states to a MAT file every ten epochs. You can then use this MAT file 
    % to resume training if it is interrupted.
    numLatentInputs = 100;


    % 5 Train the GAN. This can take multiple hours to run.
    iteration = 0;
    
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
end