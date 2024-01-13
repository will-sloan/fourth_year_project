function out = sigmoid_cross_entropy_with_logits(x,z)
out = max(x, 0) - x .* z + log(1 + exp(-abs(x)));
end

function w = iGlorotInitialize(sz)
if numel(sz) == 2
    numInputs = sz(2);
    numOutputs = sz(1);
else
    numInputs = prod(sz(1:3));
    numOutputs = prod(sz([1 2 4]));
end
multiplier = sqrt(2 / (numInputs + numOutputs));
w = multiplier * sqrt(3) * (2 * rand(sz,"single") - 1);
end

function out = preprocessAudio(in,fs)
% Ensure mono
in = mean(in,2);

% Resample to 16kHz
x = resample(in,16e3,fs);

% Cut or pad to have approximately 1-second length plus padding to ensure
% 128 analysis windows for an STFT with 256-point window and 128-point
% overlap.
y = trimOrPad(x,16513);

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

function removeRestrictiveLicence(percussivesoundsFolder,licensefilename)
%removeRestrictiveLicense Remove restrictive license

% Parse the licenses file that maps ids to license. Create a table to hold the info.
f = fileread(licensefilename);
K = jsondecode(f);
fns = fields(K);
T = table(Size=[numel(fns),4], ...
    VariableTypes=["string","string","string","string"], ...
    VariableNames=["ID","FileName","UserName","License"]);
for ii = 1:numel(fns)
    fn = string(K.(fns{ii}).name);
    li = string(K.(fns{ii}).license);
    id = extractAfter(string(fns{ii}),"x");
    un = string(K.(fns{ii}).username);
    T(ii,:) = {id,fn,un,li};
end

% Remove any files that prohibit commercial use. Find the file inside the
% appropriate folder, and then delete it.
unsupportedLicense = "http://creativecommons.org/licenses/by-nc/3.0/";
fileToRemove = T.ID(strcmp(T.License,unsupportedLicense));
for ii = 1:numel(fileToRemove)
    fileInfo = dir(fullfile(percussivesoundsFolder,"**",fileToRemove(ii)+".wav"));
    delete(fullfile(fileInfo.folder,fileInfo.name))
end

end