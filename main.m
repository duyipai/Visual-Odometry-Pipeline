%% Setup
ds = 2; % 0: KITTI, 1: Malaga, 2: parking
parking_path = './data/parking/';
if ds == 0
    % need to set kitti_path to folder containing "00" and "poses"
    assert(exist('kitti_path', 'var') ~= 0);
    ground_truth = load([kitti_path '/poses/00.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
elseif ds == 1
    % Path containing the many files of Malaga 7.
    assert(exist('malaga_path', 'var') ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    % Path containing images, depths and all...
    assert(exist('parking_path', 'var') ~= 0);
    last_frame = 598;
    K = load([parking_path '/K.txt']);
    bootstrap_frames = 0:last_frame;
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end
%% Bootstrap
% need to set bootstrap_frames
global keyPointTracker candidateTracker;
if ds == 0
    bootstrap_frames(2) = 2;
    img0 = imread([kitti_path '/00/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(1))]);
    img1 = imread([kitti_path '/00/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(2))]);
elseif ds == 1
    img0 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(1)).name]));
    img1 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(2)).name]));
elseif ds == 2
    img0 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(1))]));
    img1 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(2))]));
else
    assert(false);
end

% construct intrinsic parameter
focalLength = [K(1, 1); K(2, 2)];
principalPoint = [K(1, 3); K(2, 3)];
imageSize = size(img0);
cameraParams = cameraIntrinsics(focalLength,principalPoint,imageSize);

% init tracker
featurePoints = detectHarrisFeatures(img0);
featurePoints = featurePoints.Location;
keyPointTracker = vision.PointTracker;
initialize(keyPointTracker, featurePoints, img0);

% init landmarks
[key_points,validity] = keyPointTracker(img1);
featurePoints = featurePoints(validity, :);
key_points = key_points(validity, :);
[E, inlier_index]= estimateEssentialMatrix(featurePoints, key_points, cameraParams);
[init_rotation, init_location] = relativeCameraPose(E, cameraParams, featurePoints(inlier_index, :), key_points(inlier_index, :));
camMatrix1 = cameraMatrix(cameraParams, eye(3), zeros(1, 3));
camMatrix2 = cameraMatrix(cameraParams, init_rotation, init_location);
worldPoint = triangulate(featurePoints, key_points, camMatrix1, camMatrix2);
S_i.X = worldPoint';

featurePoints_new = detectHarrisFeatures(img1);
featurePoints_new = floor(featurePoints_new.Location + 0.5);
zero_img = zeros(size(img0));
one_img = ones(size(img0));
occupied_points = key_points;
occupied_points = floor(occupied_points + 0.5);
ind = sub2ind(size(img0), occupied_points(:, 2), occupied_points(:, 1));
one_img(ind) = 0;
ind = sub2ind(size(img0), featurePoints_new(:, 2), featurePoints_new(:, 1));
zero_img(ind) = 1;
featurePoints_new = zero_img .* one_img;
[row, col] = find(featurePoints_new);
featurePoints_new = [col, row];
candidate_points = featurePoints_new;
candidateTracker = vision.PointTracker;
initialize(candidateTracker, candidate_points, img1);% init candidate tracker
numOfFeature = size(candidate_points, 1);
S_i.F = candidate_points';
T = [init_rotation, init_location'];
S_i.T = repmat(T(:), [1, numOfFeature]);
%% Continuous operation
range = (bootstrap_frames(2)+1):last_frame;
global bearingAngleCosThreshold;
bearingAngleCosThreshold = 0.5;
rng(1);
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if ds == 0
        image = imread([kitti_path '/00/image_0/' sprintf('%06d.png',i)]);
    elseif ds == 1
        image = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
    elseif ds == 2
        image = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
    else
        assert(false);
    end
    
    [S_i, T_i] = processFrame(image, S_i, cameraParams);
    % Makes sure that plots refresh.  
    
    pause(0.01);
end
