clc; clear; close all;

%% Part(1) Texture2
% Load the image
texture2 = imread('texture2.gif');
if size(texture2, 3) == 3
    texture2 = rgb2gray(texture2);
end

% Apply Gabor filter with specified parameters
F = 0.059;          % cycles/pixel
theta = 135;        % degrees
sigma = 8;          % standard deviation
sigma_smooth = 24;  % for smoothing filter

texture2_filtered = applyGaborFilter(texture2, F, theta, sigma);
texture2_smoothed = applyGaussianSmoothing(texture2_filtered, sigma_smooth);

% Display results
figure;
subplot(2,3,1); imshow(texture2); title('Original texture2');axis on;
subplot(2,3,2); imshow(texture2_filtered, []); title('Gabor filtered (m)');axis on;
subplot(2,3,3); imshow(texture2_smoothed, []); title('Smoothed (m'')');axis on;
subplot(2,3,5);
surf(texture2_filtered, 'EdgeColor', 'none');
shading interp;
title('3D plot of m');

subplot(2,3,6);
surf(texture2_smoothed, 'EdgeColor', 'none');
shading interp;
title('3D plot of m''');

% Threshold for segmentation (adjust as needed)
thresholdValue = 0.7;
threshold_texture2 = segmentImage(texture2_smoothed,texture2, thresholdValue);
%segmented_texture2 = segmentImage(texture2_smoothed, uint8(texture2), threshold_texture2);

subplot(2,3,4);
imshow(threshold_texture2);axis on;
title(['Segmented(t = ' num2str(thresholdValue) ')']);

%% Part(2) Texture1
% SAME WITH TEXTURE1
texture1= imread('texture1.gif');
if size(texture1, 3) == 3
    texture1 = rgb2gray(texture1);
end

% Apply Gabor filter with specified parameters
F = 0.042;          % cycles/pixel
theta = 0;        % degrees
sigma = 24;          % standard deviation
sigma_smooth = 24;  % for smoothing filter

texture1_filtered = applyGaborFilter(texture1, F, theta, sigma);
texture1_smoothed = applyGaussianSmoothing(texture1_filtered, sigma_smooth);

% Display results
figure;
subplot(2,3,1); imshow(texture1); title('Original texture1');axis on;
subplot(2,3,2); imshow(texture1_filtered, []); title('Gabor filtered (m)');axis on;
subplot(2,3,3); imshow(texture1_smoothed, []); title('Smoothed (m'')');axis on;

subplot(2,3,5);
surf(texture1_filtered, 'EdgeColor', 'none');
shading interp;
title('3D plot of m(x,y)');

subplot(2,3,6);
surf(texture1_smoothed, 'EdgeColor', 'none');
shading interp;
title('3D plot of m''(x,y)');

% Threshold for segmentation (adjust as needed)
thresholdValue = 0.5;
threshold_texture1 = segmentImage(texture1_smoothed, texture1, thresholdValue);

subplot(2,3,4);
imshow(threshold_texture1);axis on;
title(['Segmented(t = ' num2str(thresholdValue) ')']);


%% Part(3) d9d77

%read img
d9d77= imread('d9d77.gif');
if size(d9d77, 3) == 3
    d9d77 = rgb2gray(d9d77);
end

% Apply Gabor filter with specified parameters
F = 0.063;          % cycles/pixel
theta = 60;        % degrees
sigma = 36;          % standard deviation
sigma_smooth = 40;  % for smoothing filter

d9d77_filtered = applyGaborFilter(d9d77, F, theta, sigma);
d9d77_smoothed = applyGaussianSmoothing(d9d77_filtered, sigma_smooth);
%d9d77_smoothed = medfilt2(d9d77_filtered,[40,40]);
%d9d77_smoothed = ordfilt2(d9d77_filtered,1600,ones(40,40));
% Display results
figure;
subplot(2,3,1); imshow(d9d77); title('Original d9d77');axis on;
subplot(2,3,2); imshow(d9d77_filtered, []); title('Gabor filtered (m)');axis on;
%subplot(2,3,3); imshow(d9d77_smoothed, []); title('Smoothed (m'')');axis on;

subplot(2,3,5);
surf(d9d77_filtered, 'EdgeColor', 'none');
shading interp;
title('3D plot of m');

%subplot(2,3,6);
%surf(d9d77_smoothed, 'EdgeColor', 'none');
%shading interp;
%title('3D plot of m''');

% Threshold for segmentation (adjust as needed)
thresholdValue = 0.3;
threshold_d9d77 = segmentImage(d9d77_filtered, d9d77, thresholdValue);

subplot(2,3,4);
imshow(threshold_d9d77);axis on;
title(['Segmented(t = ' num2str(thresholdValue) ')']);

%% Part(4) d4d29
%read img
d4d29= imread('d4d29.gif');
if size(d4d29, 3) == 3
    d4d29 = rgb2gray(d4d29);
end

% Apply Gabor filter with specified parameters
F = 0.6038;          % cycles/pixel
theta = -50.5;        % degrees
sigma = 8;          % standard deviation
sigma_smooth = 40;  % for smoothing filter



d4d29_filtered = applyGaborFilter(d4d29, F, theta, sigma);
d4d29_smoothed = applyGaussianSmoothing(d4d29_filtered, sigma_smooth);

% Display results
figure;
subplot(2,3,1); imshow(d4d29); title('Original d4d29');axis on;
subplot(2,3,2); imshow(d4d29_filtered, []); title('Gabor filtered (m)');axis on;
subplot(2,3,3); imshow(d4d29_smoothed, []); title('Smoothed (m'')');axis on;

subplot(2,3,5);
surf(d4d29_filtered, 'EdgeColor', 'none');
shading interp;
title('3D plot of m');
view(-45, 20);

subplot(2,3,6);
surf(d4d29_smoothed, 'EdgeColor', 'none');
shading interp;
title('3D plot of m''');
view(-45, 20);

% Threshold for segmentation (adjust as needed)
thresholdValue = 0.5;
threshold_d4d29 = segmentImage(d4d29_smoothed, d4d29, thresholdValue);

subplot(2,3,4);
imshow(threshold_d4d29);axis on;
title(['Segmented(t = ' num2str(thresholdValue) ')']);


%% Functions
function m = applyGaborFilter(I, F, theta, sigma)
    % Convert input to double if necessary
    if ~isa(I, 'double')
        I = double(I);
    end
    
    % Get image dimensions
    [rows, cols] = size(I);

    % Calculate filter size (4σ+1)
    filterSize = round(4*sigma) + 1;
    halfSize = floor(filterSize/2);
    
    % Create coordinate ranges
    x = -halfSize:halfSize;
    y = -halfSize:halfSize;
    
    % Convert theta to radians
    thetaRad = deg2rad(theta);

    % precompute the GEF h(x, y)
    g1 = exp(-x.^2 / (2*sigma^2));
    h1 = g1 .* exp(1i * 2 * pi * F * x * cos(thetaRad));
    
    g2 = exp(-y.^2 / (2*sigma^2));
    h2 = g2 .* exp(1i * 2 * pi * F * y * sin(thetaRad));
    
    % Calculate valid region
    valid_x_min = halfSize + 1;
    valid_x_max = cols - halfSize;
    valid_y_min = halfSize + 1;
    valid_y_max = rows - halfSize;
    
    % Initialize intermediate and output images
    i1 = zeros(rows, cols);
    i2 = zeros(rows, cols, 'like', 1i); % Complex output

    % Initialize output
    validRows = (1+halfSize):(rows-halfSize);
    validCols = (1+halfSize):(cols-halfSize);
    m = zeros(length(validRows), length(validCols));
    
    % Step 1: Convolve with h1(x) in x-direction
    for y = valid_y_min:valid_y_max
        for x = valid_x_min:valid_x_max
            sum_val = 0;
            for x_prime = -halfSize:halfSize
                idx = x_prime + halfSize + 1; % Index into h1
                sum_val = sum_val + I(y, x-x_prime) * h1(idx);
            end
            i1(y, x) = sum_val;
        end
    end
    
    % Step 2: Convolve with h2(y) in y-direction
    for y = valid_y_min:valid_y_max
        for x = valid_x_min:valid_x_max
            sum_val = 0;
            for y_prime = -halfSize:halfSize
                idx = y_prime + halfSize + 1; % Index into h2
                sum_val = sum_val + i1(y-y_prime, x) * h2(idx);
            end
            i2(y, x) = sum_val;
        end
    end
    
    % Step 3: Take magnitude
    m = abs(i2(valid_y_min:valid_y_max, valid_x_min:valid_x_max));
end

function smoothed  = applyGaussianSmoothing(inputImage, sigma)    
    % Create Gaussian kernel
    filterSize = round(4*sigma) + 1;  % Kernel size
    halfSize = floor(filterSize/2);
    [x, y] = meshgrid(-halfSize:halfSize, -halfSize:halfSize);
    g = exp(-(x.^2 + y.^2)/(2*sigma^2));
    g = g / sum(g(:));  % Normalize
    
    % Determine valid region
    [rows, cols] = size(inputImage);
    validRows = (1+halfSize):(rows-halfSize);
    validCols = (1+halfSize):(cols-halfSize);
    
    % Initialize output
    smoothed = zeros(length(validRows), length(validCols));
    
    % Perform convolution only in valid region
    for i = 1:length(validRows)
        for j = 1:length(validCols)
            % Extract neighborhood
            patch = inputImage(validRows(i)-halfSize:validRows(i)+halfSize, validCols(j)-halfSize:validCols(j)+halfSize);
            % Apply Gaussian
            smoothed(i,j) = sum(sum(patch .* g));
        end
    end
end

% Function to segment the image based on thresholding
function segmentedOverlay = segmentImage(filtered, original, thresholdValue)
    % Apply thresholding only to valid regions (non-zero)
    % Normalize image to [0 1] range if needed
    if max(filtered(:)) > 1
        filtered = mat2gray(filtered);
    end
    binaryMask = filtered > thresholdValue;
    
    % Make sure original and filtered are the same size
    [filtered_rows, filtered_cols] = size(filtered);
    [orig_rows, orig_cols, orig_channels] = size(original);
    
    % Crop or resize the original image to match filtered dimensions
    if orig_rows ~= filtered_rows || orig_cols ~= filtered_cols
        % Check if we need to crop original (it's larger)
        if orig_rows >= filtered_rows && orig_cols >= filtered_cols
            % Calculate center offsets for cropping
            row_offset = floor((orig_rows - filtered_rows) / 2);
            col_offset = floor((orig_cols - filtered_cols) / 2);
            
            % Crop the original image to match filtered size
            if orig_channels == 1
                originalCropped = original(row_offset+1:row_offset+filtered_rows, ...
                                          col_offset+1:col_offset+filtered_cols);
            else
                originalCropped = original(row_offset+1:row_offset+filtered_rows, ...
                                          col_offset+1:col_offset+filtered_cols, :);
            end
        else
            % If original is smaller (unlikely but possible), pad it
            originalCropped = zeros(filtered_rows, filtered_cols, orig_channels, 'like', original);
            row_offset = floor((filtered_rows - orig_rows) / 2);
            col_offset = floor((filtered_cols - orig_cols) / 2);
            
            if orig_channels == 1
                originalCropped(row_offset+1:row_offset+orig_rows, ...
                               col_offset+1:col_offset+orig_cols) = original;
            else
                originalCropped(row_offset+1:row_offset+orig_rows, ...
                               col_offset+1:col_offset+orig_cols, :) = original;
            end
        end
    else
        originalCropped = original;
    end
    
    % Ensure original is uint8
    if ~isa(originalCropped, 'uint8')
        originalCropped = uint8(originalCropped);
    end
    
    % Create RGB version of original cropped image
    if size(originalCropped, 3) == 1
        originalRGB = cat(3, originalCropped, originalCropped, originalCropped);
    else
        originalRGB = originalCropped;
    end
    
    % Create a visualization with the segmentation overlaid on the original
    segmentedOverlay = originalRGB;
    
    % Add colored overlay for segmented regions
    for x = 1:filtered_rows
        for y = 1:filtered_cols
            if binaryMask(x,y)
                segmentedOverlay(x,y,1) = min(255, originalRGB(x,y,1) + 80);
                segmentedOverlay(x,y,2) = max(0, originalRGB(x,y,2) - 30);
                segmentedOverlay(x,y,3) = max(0, originalRGB(x,y,3) - 30);
            end
        end
    end
end
