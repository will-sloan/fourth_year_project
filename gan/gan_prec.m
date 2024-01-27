function y = synthesizePercussiveSound

    persistent pGeneratorParameters pMean pSTD
    if isempty(pGeneratorParameters)
        % If the MAT file does not exist, download it
        filename = "drumGeneratorWeights.mat";
        load(filename,"SMean","SStd","generatorParameters");
        pMean = SMean;
        pSTD  = SStd;
        pGeneratorParameters = generatorParameters;
    end
    
    % Generate random vector
    dlZ = dlarray(2 * ( rand(1,1,100,1,"single") - 0.5 ));
    
    % Generate spectrograms
    dlXGenerated = modelGenerator(dlZ,pGeneratorParameters);
    
    % Convert from dlarray to single
    S = dlXGenerated.extractdata;
    
    S = S.';
    % Zero-pad to remove edge effects
    S = [S ; zeros(1,128)];
    
    % Reverse steps from training
    S = S * 3;
    S = (S.*pSTD) + pMean;
    S = exp(S);
    
    % Make it two-sided
    S = [S ; S(end-1:-1:2,:)];
    % Pad with zeros at end and start
    S = [zeros(256,100) S zeros(256,100)];
    
    % Reconstruct the signal using a fast Griffin-Lim algorithm.
    myAudio = stftmag2sig(S,256, ...
        FrequencyRange="twosided", ...
        Window=hann(256,"periodic"), ...
        OverlapLength=128, ...
        MaxIterations=20, ...
        Method="fgla");
    myAudio = myAudio./max(abs(myAudio),[],"all");
    y = myAudio(128*100:end-128*100);
end