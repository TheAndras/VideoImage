
currentdirectory = pwd;
disp(pwd)
video=VideoReader(strcat(currentdirectory,'\vidiR.mov'));
pathname = fileparts(strcat(currentdirectory,'\imgVidi\'));

% Load video & save frames.
k=1;
img=10000;
% Image enchance method, but not really works :D 
% LEN = 11;
% THETA = 6;
% PSF = fspecial('motion', LEN, THETA);
for i = 1:video.NumberOfFrames
    k=k+1;
    if k==2
        baseFileName=strcat('frame',num2str(img),'.jpg');
        fullFileName = fullfile(pathname, baseFileName); 
        frame = read(video, i);   
%         enhancedFrame = deconvwnr(frame, PSF, 0.1);
        imwrite(frame,fullFileName);
        img=img+1;
        k=1;
    end
end

buildingDir = pathname;
buildingScene = imageDatastore(buildingDir);

% Display saved frames
montage(buildingScene.Files)




% Read the first frame.
I = readimage(buildingScene, 1);

% Initialize features for I(1)
grayImage = rgb2gray(I);
points = detectSURFFeatures(grayImage, 'MetricThreshold', 1);
[features, points] = extractFeatures(grayImage, points);

% Initialize all the transforms to the identity matrix.
numImages = numel(buildingScene.Files);
tforms(numImages) = affine2d(eye(3));

% Initialize variable to hold image sizes.
imageSize = zeros(numImages,2);
n=2;

% Iterate over remaining image pairs
while (n <= numImages) 
    
    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;

    % Read I(n).
    I = readimage(buildingScene, n);

    % Convert image to grayscale.
    grayImage = rgb2gray(I);

    % Save image size.
    imageSize(n,:) = size(grayImage);

    % Detect and extract SURF features for I(n).
    points = detectSURFFeatures(grayImage);
    [features, points] = extractFeatures(grayImage, points);

    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true, 'MatchThreshold',100);

    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);

    % Estimate the transformation between I(n) and I(n-1).
    
    if size(indexPairs, 1) >=15
           tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
           'similarity', 'Confidence', 99, 'MaxNumTrials', 2000);
        
%     Just for test.
    disp('Összerakas')   
    disp(n)
        
        tforms(n).T = tforms(n).T * tforms(n-1).T;
    
    n=n+1;
%     Just for test.
    else        
    disp('Törlés')   
    disp(n)
%   Another test method, you can see the number of the pairs. I needed this
%   one to decide about the minimal pair number.
%     disp(size(indexPairs, 1))

    buildingScene.Files = setdiff(buildingScene.Files,buildingScene.Files{n,1});
    numImages=numImages-1;

    end

 
end



% Compute the output limits  for each transform
for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end



avgXLim = mean(xlim, 2);

[~, idx] = sort(avgXLim);

centerIdx = floor((numel(tforms)+1)/2);

centerImageIdx = idx(centerIdx);




Tinv = invert(tforms(centerImageIdx));

for i = 1:numel(tforms)
    tforms(i).T = tforms(i).T * Tinv.T;
end




for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of final image.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" image.
videoImage = zeros([height width 3], 'like', I);



blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the final image.
for i = 1:numImages

    I = readimage(buildingScene, i);

    % Transform I into the final image.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);

    % Generate a binary mask.
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);

    % Overlay the warpedImage onto the image.
    videoImage = step(blender, videoImage, warpedImage, mask);
end

figure
imwrite(videoImage, 'videoImage.jpg')
imshow(videoImage)
