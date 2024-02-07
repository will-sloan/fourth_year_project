filename = 'sofa_hrtfs/RIEC_hrir_subject_001.sofa';
s = sofaread(filename);

% Get all zero indexed source positions
elevation = s.SourcePosition(:,2);
flat_recordings = find(elevation == 0);

% Get all the flat hrirs
% Now we got our baseline

hrir_desc = s.SourcePosition(flat_recordings,:);
hrirs = s.Numerator(flat_recordings,:,:); % [num, channel, sequence]

% Initialize a 5D array for the masked HRIRs
hrirs_masked = zeros([size(hrirs) 10]);

% Loop over each HRIR
for i = 1:size(hrirs, 1)
    % Loop over each channel
    for j = 1:size(hrirs, 2)
        % Get the current HRIR
        firFilter = squeeze(hrirs(i,j,:));
        
        % Loop over each mask
        for k = 1:10
            % Create a mask
            mask = rand(size(firFilter)) > 0.1;  % 10% of values will be masked

            % Apply the mask
            hrirs_masked_temp = firFilter;
            hrirs_masked_temp(mask) = -inf;
            
            % Store the masked HRIR in the 5D array
            hrirs_masked(i,j,:,k) = hrirs_masked_temp;
        end
    end
end