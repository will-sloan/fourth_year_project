
% Channel 1
[data, sample_rate] = audioread("..\Data\SplitDataChunks\90Deg_EARS_1_1SecChunks\90Deg_EARS_1_8500.wav");

% Time difference between samples
time_delta = 1/sample_rate;

% Make it so stft evaluates frequencies in intervals of 100 Hz
fft_length = sample_rate/100;

% Window length to give a precision of 10ms (0.01 seconds), 100Hz
window_length = 0.01/time_delta;

[stft_arr, freqs, times] = stft(data, seconds(time_delta), ...
    FFTLength=fft_length, Window=hann(window_length,"periodic"), ...
    OverlapLength=window_length/2,FrequencyRange="onesided");
%{
stft(data, seconds(time_delta), ...
    FFTLength=fft_length, Window=hann(window_length,"periodic"), ...
    OverlapLength=window_length/2,FrequencyRange="onesided");
%}

% ===============================================

% Channel 2
[data2, sample_rate2] = audioread("..\Data\SplitDataChunks\90Deg_EARS_2_1SecChunks\90Deg_EARS_2_8500.wav");

% Time difference between samples
time_delta2 = 1/sample_rate2;

% Make it so stft evaluates frequencies in intervals of 100 Hz
fft_length2 = sample_rate2/100;

% Window length to give a precision of 10ms (0.01 seconds), 100Hz
window_length2 = 0.01/time_delta2;

[stft_arr2, freqs2, times2] = stft(data2, seconds(time_delta2), ...
    FFTLength=fft_length2, Window=hann(window_length2,"periodic"), ...
    OverlapLength=window_length2/2,FrequencyRange="onesided");

%{
stft(data2, seconds(time_delta2), ...
    FFTLength=fft_length2, Window=hann(window_length2,"periodic"), ...
    OverlapLength=window_length2/2,FrequencyRange="onesided");
%}


diff = data - data2
total_diff = sum(diff)
[variance, mean_diff] = var(diff)
std_dev = sqrt(variance)


dlZ = dlarray(2 * ( rand(1,1,100,1,"single") - 0.5 ));

%{
classdef fcont
    methods
        function try_this = tryThis(x)
        
            try_this = x*2
        
        end

    end
end
%}



