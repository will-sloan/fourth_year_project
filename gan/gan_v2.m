train_model

% ================ Utility Functions ================
function out = sigmoid_cross_entropy_with_logits(x,z)
    out = max(x, 0) - x .* z + log(1 + exp(-abs(x)));
end


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


function out = preprocessAudio(in)
    % Ensure mono
    in = mean(in,2);
    
    % Resample to 16kHz
    % x = resample(in,16e3,fs);
    
    % Cut or pad to have approximately 1-second length plus padding to ensure
    % 128 analysis windows for an STFT with 256-point window and 128-point
    % overlap.
    % y = trimOrPad(x,16513);
    y = trimOrPad(in, 32000);
    
    % Normalize
    out = y./max(abs(y));
end


function y = trimOrPad(x,n)
    %trimOrPad Trim or pad audio
    %
    % y = trimOrPad(x,n) trims or pads the input x to n samples along the first
    % dimension. If x is trimmed, it is trimmed equally on the front and back.
    % If x is padded, it is padded equally on the front and back with zeros.
    % For odd-length trimming or padding, the extra sample is trimmed or padded
    % from the back.
    
    a = size(x,1);
    if a < n
        frontPad = floor((n-a)/2);
        backPad = n - a - frontPad;
        y = [zeros(frontPad,size(x,2),like=x);x;zeros(backPad,size(x,2),like=x)];
    elseif a > n
        frontTrim = floor((a-n)/2) + 1;
        backTrim = a - n - frontTrim;
        y = x(frontTrim:end-backTrim,:);
    else
        y = x;
    end
end


% ================ Generator Code ================
% Generator Architecture
function dlYGen = modelGenerator(dlX,parameters)

    % size_dlZ_init = size(dlX)

    dlYGen = fullyconnect(dlX,parameters.FC.Weights,parameters.FC.Bias,Dataformat="SSCB");
    
    % size_Gen_FC = size(dlYGen)

    dlYGen = reshape(dlYGen,[1024 16 4 size(dlYGen,2)]);

    % size_Gen_Reshape = size(dlYGen)

    dlYGen = permute(dlYGen,[3 2 1 4]);

    % size_Gen_permuted = size(dlYGen)

    % dlYGen = relu(dlYGen);
    dlYGen = leakyrelu(dlYGen, 0.2);
    
    dlYGen = dltranspconv(dlYGen,parameters.Conv1.Weights,parameters.Conv1.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    % dlYGen = relu(dlYGen);
    dlYGen = leakyrelu(dlYGen, 0.2);

    
    % size_Gen_TC1 = size(dlYGen)

    dlYGen = dltranspconv(dlYGen,parameters.Conv2.Weights,parameters.Conv2.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    % dlYGen = relu(dlYGen);
    dlYGen = leakyrelu(dlYGen, 0.2);

    
    % size_Gen_TC2 = size(dlYGen)

    dlYGen = dltranspconv(dlYGen,parameters.Conv3.Weights,parameters.Conv3.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    % dlYGen = relu(dlYGen);
    dlYGen = leakyrelu(dlYGen, 0.2);
    
    % size_Gen_TC3 = size(dlYGen)

    dlYGen = dltranspconv(dlYGen,parameters.Conv4.Weights,parameters.Conv4.Bias,Stride=2,Cropping="same",DataFormat="SSCB");
    % dlYGen = relu(dlYGen);
    dlYGen = leakyrelu(dlYGen, 0.2);

    
    % size_Gen_TC4 = size(dlYGen)

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


% ================ Discriminator Code ================
% Discriminator Architecture
function dlYDisc = modelDiscriminator(dlX,parameters)

    % size_dlX_init = size(dlX)

    dlYDisc = dlconv(dlX,parameters.Conv1.Weights,parameters.Conv1.Bias,Stride=2,Padding="same");
    % dlYDisc = dlconv(dlX,parameters.Conv1.Weights,parameters.Conv1.Bias,Stride=[2, 8], Padding=[2, 2]);
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    % size_dlYDisc_1 = size(dlYDisc)

    dlYDisc = dlconv(dlYDisc,parameters.Conv2.Weights,parameters.Conv2.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    % size_dlYDisc_2 = size(dlYDisc)

    dlYDisc = dlconv(dlYDisc,parameters.Conv3.Weights,parameters.Conv3.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    % size_dlYDisc_3 = size(dlYDisc)

    dlYDisc = dlconv(dlYDisc,parameters.Conv4.Weights,parameters.Conv4.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
    
    % size_dlYDisc_4 = size(dlYDisc)

    dlYDisc = dlconv(dlYDisc,parameters.Conv5.Weights,parameters.Conv5.Bias,Stride=2,Padding="same");
    dlYDisc = leakyrelu(dlYDisc,0.2);
     
    % size_dlYDisc_5 = size(dlYDisc)

    dlYDisc = stripdims(dlYDisc);

    % size_dlYDiscStripped = size(dlYDisc)

    dlYDisc = permute(dlYDisc,[3 2 1 4]);

    % size_dlYDiscPermuted = size(dlYDisc)

    dlYDisc = reshape(dlYDisc,(4*4*16*2)*(64*2),numel(dlYDisc)/((4*4*16*2)*(64*2)));
    
    % size_discReshape = size(dlYDisc)

    weights = parameters.FC.Weights;
    bias = parameters.FC.Bias;
    dlYDisc = fullyconnect(dlYDisc,weights,bias,Dataformat="CB");
end


% Discriminator Gradients
function gradientsDiscriminator = modelDiscriminatorGradients(discriminatorParameters,generatorParameters,X,Z)
    % Calculate the predictions for real data with the discriminator network.
    Y = modelDiscriminator(X,discriminatorParameters);
    
    % disp("Model discriminator ran!");

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
    weights = iGlorotInitialize([1, (8 * 16) * (dim * 8)]);
    bias = zeros(1,1,"single");
    discriminatorParameters.FC.Weights = dlarray(weights);
    discriminatorParameters.FC.Bias = dlarray(bias);
end


% Get training the data
function training_data = getTrainingData(angle)

    % 0 Load Training Data
    % path = sprintf("..\\Data\\TestData\\%dDeg_EARS_1", angle);
    % F:\DuncanM\Matlab\Data\TrainingData\TIMITOnly\90Deg_Leading500
    % path = sprintf("..\\Data\\TrainingData\\TIMITOnly\\%dDeg_Leading500", angle);
    % F:\DuncanM\Matlab\Data\TrainingData\TIMITOnly\90Deg_EARS_1
    path = sprintf("..\\Data\\TrainingData\\TIMITOnly\\%dDeg_EARS_1", angle);

    canUseParallelPool = true;
    

    % Change this to match our input data
    ads = audioDatastore(path,IncludeSubfolders=true);

    % 1 Define the STFT parameters.

    assumed_sample_rate = 32000;

    frequency_step = 62.5; % Frequency precision
    time_precision = 1/frequency_step; % Time precision
    

    fft_length = assumed_sample_rate/frequency_step;
    win_length = assumed_sample_rate * time_precision;
    win = hann(win_length,"periodic");

    total_time = 1; % total time of a sample in seconds
    target_time_points = 128;
    
    overlap_portion = ((total_time*frequency_step)-1)/(target_time_points-1);
    overlap_length = ceil( win_length * (1 - overlap_portion) );
    
    % 2 First, determine the number of partitions for the dataset. If 
    % you do not have Parallel Computing Toolbox™, use a single partition.
    if canUseParallelPool
        pool = gcp;
        numPar = numpartitions(ads,pool);
    else
        numPar = 1;
    end

    numParMsg = sprintf("numPar = %d\n", numPar);
    disp(numParMsg);

    % 3 For each partition, read from the datastore and compute the STFT.
    parfor ii = 1:numPar
    
        subds = partition(ads,numPar,ii);
        % subds = ads;
        % STrain = zeros(fft_length/2+1,128,1,numel(subds.Files));
        % STrain = zeros(fft_length/2+1,199,1,numel(subds.Files));
        STrain = zeros(fft_length,target_time_points,1,numel(subds.Files));
        
        % size(STrain)
    
        for idx = 1:numel(subds.Files)
            if mod(idx, 100) == 0
                ack = sprintf("Iteration %d", idx)
            end
    
            % Read audio
            [x,xinfo] = read(subds);
    
            % Preprocess
            x = preprocessAudio(single(x));
    
            % STFT
            S0 = stft(x, seconds(1/xinfo.SampleRate), ...
                FFTLength = fft_length, Window=win, ...
                OverlapLength=overlap_length, FrequencyRange="centered");
            
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

    checkSize = size(STrain)

    % 9 Discard the last frequency bin to force the number of STFT bins 
    % to a power of two (which works well with convolutional layers).
    % STrain = STrain(1:end-1,:,:,:);

    % 10 Permute the dimensions in preparation for feeding to the 
    % discriminator.
    STrain = permute(STrain,[2 1 3 4]);

    training_data = STrain;
end


function sample = sampleOutputAndLoss(generatorParameters, discriminatorParameters, X, Z)

    % Run another GAN game to get a sample of results + loss for this
    % iteration, averaged out over the epoch
    
    Y = modelDiscriminator(X, discriminatorParameters);

    Xgen = modelGenerator(Z,generatorParameters);
    Ygen = modelDiscriminator(dlarray(Xgen,"SSCB"),discriminatorParameters);

    % Calculate the GAN loss.
    lossDiscriminator = ganDiscriminatorLoss(Y,Ygen);
    lossGenerator = ganGeneratorLoss(Ygen);

    sample = {Xgen, Y, Ygen, lossGenerator, lossDiscriminator};
end


% Training the model
function train_model

    % First, specify training options

    % 1 Train with a mini-batch size of 64 for 1000 epochs.
    % maxEpochs = 1000;
    % miniBatchSize = 64;
    maxEpochs = 4;
    miniBatchSize = 64;

    saveFrequency = 2;


    test_angle = 90;
    STrain = getTrainingData(test_angle);

    % 2 Compute the number of iterations required to consume the data.

    % numIterationsPerEpoch = floor(size(STrain,4)/miniBatchSize)
    numIterationsPerEpoch = floor(size(STrain,4)/miniBatchSize);

    % numIterationsPerEpochProper = floor(size(STrain,4)/miniBatchSize)

    % Pretty much add a breakpoint here
    % k = waitforbuttonpress;

    % 3 Specify the options for Adam optimization. Set the learn rate of 
    % the generator and discriminator to 0.0002. For both networks, use a 
    % gradient decay factor of 0.5 and a squared gradient decay factor of 0.999.
    learnRateGenerator = 0.0002;
    learnRateDiscriminator = 0.0002;
    gradientDecayFactor = 0.5;
    squaredGradientDecayFactor = 0.999;

    % 4 Train on a GPU if one is available. Using a GPU requires 
    % Parallel Computing Toolbox™.
    executionEnvironment = "auto";
    canUseGPU = true;

    loadData = true;

    % 5 Initialize the generator and discriminator weights. The 
    % initializeGeneratorWeights and initializeDiscriminatorWeights 
    % functions return random weights obtained using Glorot uniform 
    % initialization. The functions are included at the end of this example.
    generatorParameters = initializeGeneratorWeights;
    discriminatorParameters = initializeDiscriminatorWeights;

    if loadData

        disp("Loading saved data!");
        savedData = load('audiogancheckpoint9.mat');

        generatorParameters = savedData.generatorParameters;
        discriminatorParameters = savedData.discriminatorParameters;

    end
    

    % GAN

    % 1 Initialize the parameters for Adam.
    trailingAvgGenerator = [];
    trailingAvgSqGenerator = [];
    trailingAvgDiscriminator = [];
    trailingAvgSqDiscriminator = [];


    samplesPerEpoch = 10;
    samplePeriod = floor(numIterationsPerEpoch/samplesPerEpoch)
    sampleIndex = 1;
    

    numSamples = floor(maxEpochs/saveFrequency);

    epochGeneratorSample = zeros(128, 512, numSamples);
    epochDiscriminatorSample = zeros(1, 2, numSamples);
    epochAverageLosses = zeros(1, 2, numSamples);

    tempGeneratorSample = zeros(128, 512, samplesPerEpoch);
    tempDiscriminatorSample = zeros(1, 2, samplesPerEpoch);
    tempAverageLosses = zeros(1, 2, samplesPerEpoch);

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
            % Z = 2 * ( rand(1,1,numLatentInputs,miniBatchSize,"single") - 0.5 ) ;
            Z = single(normrnd(0, 1, 1,1,numLatentInputs,miniBatchSize));
            dlZ = dlarray(Z);
    
            % If training on a GPU, then convert data to gpuArray.
            if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
                % disp("Using GPU!");
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
        
        
            if (mod(index, samplePeriod) == 0)
                % tempSample = sampleOutputAndLoss(generatorParameters, discriminatorParameters, dlX, dlZ);
    
                tempSample = dlfeval(@sampleOutputAndLoss, generatorParameters, ... 
                discriminatorParameters, dlX, dlZ);
    
                %{
                testSize = size(tempSample{1})
                testSize2 = size(tempSample{2})
                testSize3 = size(tempSample{3})
                testSize4 = size(tempSample{4})
                testSize5 = size(tempSample{5})
                %}
    
                tempGeneratorSample(:, :, sampleIndex) = mean(dlarray(tempSample{1}, "SSCB"), 4);
                tempDiscriminatorSample(1, 1, sampleIndex) = mean(dlarray(tempSample{2}, "SB"), 2);
                tempDiscriminatorSample(1, 2, sampleIndex) = mean(dlarray(tempSample{3}, "SB"), 2);
                tempAverageLosses(1, 1, sampleIndex) = tempSample{4};
                tempAverageLosses(1, 2, sampleIndex) = tempSample{5};
    
                sampleIndex = sampleIndex + 1;

            end
        
        end


        % Get epoch samples
        epochGeneratorSample(:, :, epoch) = mean(tempGeneratorSample, 3);
        epochDiscriminatorSample(:, :, epoch) = mean(tempDiscriminatorSample, 3);
        epochAverageLosses(:, :, epoch) = mean(tempAverageLosses, 3);

        % Every 10 epochs, save a training snapshot to a MAT file.
        if mod(epoch,saveFrequency)==0
            disp("Epoch " + epoch + " out of " + maxEpochs + " complete.");
            if saveCheckpoints
                
                saveIter = int32(epoch/saveFrequency);
                paramSaveFilename = sprintf("audiogancheckpoint_temp%d.mat", saveIter);

                disp(paramSaveFilename);

                % Save checkpoint in case training is interrupted.
                save(paramSaveFilename, ...
                    "generatorParameters","discriminatorParameters", ...
                    "trailingAvgDiscriminator","trailingAvgSqDiscriminator", ...
                    "trailingAvgGenerator","trailingAvgSqGenerator","iteration", ...
                    "epochGeneratorSample", "epochDiscriminatorSample", ...
                    "epochAverageLosses");
            end
        end

        %{
        testSizeGenSample = size(mean(tempGeneratorSample, 3))
        testSizeDiscSample = size(mean(tempDiscriminatorSample, 3))
        testSizeLossSample = size(mean(tempAverageLosses, 3))
        %}
    
        msg = sprintf("Epoch %d done!", epoch);
        disp(msg);
    end
end