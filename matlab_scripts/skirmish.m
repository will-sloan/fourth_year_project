h = readhrtf(0, 0, 'L');
h = reshape(h, [1, 2, 512]);

pluhL = [readhrtf(0, 0, 'L'); 
    readhrtf(20, 0, 'L'); ... 
    readhrtf(40, 0, 'L'); readhrtf(60, 0, 'L')];


pluhR = [readhrtf(0, 0, 'R'); 
    readhrtf(20, 0, 'R'); ... 
    readhrtf(40, 0, 'R'); readhrtf(60, 0, 'R')];

pluh = [pluhL(1, :), pluhR(1, :); pluhL(2, :), pluhR(2, :); ...
    pluhL(3, :), pluhR(3, :); pluhL(4, :), pluhR(4, :)];

% size(pluh)
pluh = reshape(pluh, [4, 2, 512]);

% size(pluh)

source = [0,0;20,0;40,0;60,0];

des = [30, 0];

size(source)

result = interpolateHRTF(pluh, source, des, Algorithm="vbap");


actualL = readhrtf(30, 0, 'L');
actualR = readhrtf(30, 0, 'R');

actual = [actualL(1, :); actualR(1, :)];
