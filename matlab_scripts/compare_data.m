angles = [0, 15, 30, 45, 60, 75, 90];

data_stats = zeros(length(angles), 3)

%{
disp("The FUCK");
data_stats(1, :)
disp("Is this??");
%}

format("shortG")

for i = 1:length(angles)
    data_stats(i, :) = compare(angles(i), 171);

end

my_stft = graph_stft(30, 171);

% disp("Yooooooo");
% my_stft_size = size(my_stft)

% differences = compare2(90, 171);

% res = tryThis(4);


sample_angle = 30;
sample_chunk = 171;
path = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_1\\%dDeg_EARS_1_%d.wav", sample_angle, sample_angle, sample_chunk)
path2 = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_2\\%dDeg_EARS_2_%d.wav", sample_angle, sample_angle, sample_chunk);

if sample_angle == 0 || sample_angle == 15
    path = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_1\\%dDeg_EARSFullAudioRecording_1_%d.wav", sample_angle, sample_angle, sample_chunk)
    path2 = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_2\\%dDeg_EARSFullAudioRecording_2_%d.wav", sample_angle, sample_angle, sample_chunk);
end

% Channel 1
[data, sample_rate] = audioread(path);

% Channel 2
[data2, sample_rate2] = audioread(path2);


% frequency step = 100Hz of precision
freq_step = 100;

% time precision = 10ms = 0.01s
time_precision = 0.01;

% Time difference between samples
time_delta = 1/sample_rate;

% Make it so stft evaluates frequencies in intervals of 100 Hz
fft_length = sample_rate/freq_step;

% Window length to give a precision of 10ms (0.01 seconds), 100Hz
window_length = time_precision/time_delta;

[my_stft, freqs] = stft(data, seconds(time_delta), ...
FFTLength=fft_length, Window=hann(window_length,"periodic"), ...
OverlapLength=window_length/2, FrequencyRange="centered");



function stats = compare(angle, chunk_no)
    path = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_1\\%dDeg_EARS_1_%d.wav", angle, angle, chunk_no)
    path2 = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_2\\%dDeg_EARS_2_%d.wav", angle, angle, chunk_no);
    
    if angle == 0 || angle == 15
        path = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_1\\%dDeg_EARSFullAudioRecording_1_%d.wav", angle, angle, chunk_no)
        path2 = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_2\\%dDeg_EARSFullAudioRecording_2_%d.wav", angle, angle, chunk_no);
    end
    
    % Channel 1
    [data, sample_rate] = audioread(path);
    
    % Channel 2
    [data2, sample_rate2] = audioread(path2);
    
    
    
    diff = abs(data) - abs(data2);

    total_diff = sum(diff);
    [variance, mean_diff] = var(diff);
    std_dev = sqrt(variance);
    
    stats = [total_diff, mean_diff, std_dev];
end


function stats = compare2(angle, chunk_no)
    path = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_1\\%dDeg_EARS_1_%d.wav", angle, angle, chunk_no)
    path2 = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_2\\%dDeg_EARS_2_%d.wav", angle, angle, chunk_no);
    
    if angle == 0 || angle == 15
        path = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_1\\%dDeg_EARSFullAudioRecording_1_%d.wav", angle, angle, chunk_no)
        path2 = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_2\\%dDeg_EARSFullAudioRecording_2_%d.wav", angle, angle, chunk_no);
    end
    
    % Channel 1
    [data, sample_rate] = audioread(path);
    
    % Channel 2
    [data2, sample_rate2] = audioread(path2);
    
    
    
    diff = abs(data) - abs(data2);
    
    stats = diff;
end


function get_stft_graph = graph_stft(angle, chunk_no)

    path = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_1\\%dDeg_EARS_1_%d.wav", angle, angle, chunk_no);
    path2 = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_2\\%dDeg_EARS_2_%d.wav", angle, angle, chunk_no);
    
    if angle == 0 || angle == 15
        path = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_1\\%dDeg_EARSFullAudioRecording_1_%d.wav", angle, angle, chunk_no);
        path2 = sprintf("..\\Data\\SplitDataChunks\\%dDeg_EARS_2\\%dDeg_EARSFullAudioRecording_2_%d.wav", angle, angle, chunk_no);
    end

    % Channel 1
    [data, sample_rate] = audioread(path);
    
    % Channel 2
    [data2, sample_rate2] = audioread(path2);


    % frequency step = 100Hz of precision
    freq_step = 100;

    % time precision = 10ms = 0.01s
    time_precision = 0.01;

    % Time difference between samples
    time_delta = 1/sample_rate;

    % Make it so stft evaluates frequencies in intervals of 100 Hz
    fft_length = sample_rate/freq_step;
    
    % Window length to give a precision of 10ms (0.01 seconds), 100Hz
    window_length = time_precision/time_delta;

    [get_stft_graph, freqs] = stft(data, seconds(time_delta), ...
    FFTLength=fft_length, Window=hann(window_length,"periodic"), ...
    OverlapLength=window_length/2, FrequencyRange="centered");

    % fft_length
    % size(freqs)

end


function how_dis_work

    ads = audioDatastore(percussivesoundsFolder,IncludeSubfolders=true);

end