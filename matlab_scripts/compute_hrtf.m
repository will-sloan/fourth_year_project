
angle = 90;
chunkNo = 1434;
path = sprintf("../../Data/44.1kHz/SplitDataChunks/Original_Adj/Original_%d.wav", chunkNo);

output = get_hrtf(path, 0, angle);

% Function to compute the HRTF of an input audio file
% Final output size will be (2 x M), where M is the number of samples
% in the input audio
function output_hrtf = get_hrtf(input_audio_file, elev, angle)

    % Left channel HRIR
    hL_Raw = readhrtf(elev, angle, 'L');
    hL = hL_Raw(1, :);

    % Right channel HRIR
    hR_Raw = readhrtf(elev, angle, 'R');
    hR = hR_Raw(1, :);
    
    % Read in the input audio chunk
    audioChunk = audioread(input_audio_file);
    
    % Swap the array dimensions to make the number of samples equal
    % to the number of rows
    inputData = permute(audioChunk, [2, 1]);

    input_size = size(inputData(1, :));
    num_samples = input_size(2);
    
    % Compute spatialized audio in the left and right ears
    outputL = conv(hL, inputData);
    outputR = conv(hR, inputData);

    % Resulting size of the output data
    result_length = num_samples+512-1;
    
    
    % Discard the first 512 outputs (The HRIR is 512 samples long)
    output_hrtf = [outputL(:, 512:result_length); outputR(:, 512:result_length)];

end


% Function for grabbing and viewing the stft of an audio clip
function get_stft_graph = graph_stft(data, sample_rate)

    % frequency step = 62.5Hz of precision
    freq_step = 62.5;
    time_precision = 1/freq_step;


    % Time difference between samples
    time_delta = 1/sample_rate;

    % Make it so stft evaluates frequencies in intervals of 100 Hz
    fft_length = ceil(sample_rate/freq_step);
    
    % Window length to give a precision of 10ms (0.01 seconds), 100Hz
    window_length = time_precision/time_delta;

    overlap_length = ceil(window_length/2);

    [get_stft_graph, freqs] = stft(data, seconds(time_delta), ...
    FFTLength=fft_length, Window=hann(ceil(window_length),"periodic"), ...
    OverlapLength=overlap_length, FrequencyRange="centered");


    % stft(...) with no assignment on the left-hand side will plot the
    % result. Comment this bit out if you do not want the plot
    stft(data, seconds(time_delta), ...
    FFTLength=fft_length, Window=hann(ceil(window_length),"periodic"), ...
    OverlapLength=overlap_length, FrequencyRange="centered");

end
