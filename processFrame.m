function [S_i, T_i] = processFrame(I_i, S_prev, cameraParams) % remember to use spacial coordinate
    global bearingAngleCosThreshold;
    
    % first extract current pose
    global keyPointTracker candidateTracker;
    [key_points,validity] = keyPointTracker(I_i);
    key_points = key_points(validity, :);
    S_prev.X = S_prev.X(:, validity);
    [worldOrientation,worldLocation] = estimateWorldCameraPose(key_points, S_prev.X', cameraParams);
    T_i = [worldOrientation, worldLocation];
    
    % then seek new tracking points from candidates
    [candidate_points, validity] = candidateTracker(I_i);
    candidate_points = candidate_points(validity, :);
    S_prev.F = S_prev.F(:, validity);
    S_prev.T = S_prev.T(:, validity);
    canBeAdded = zeros(1, size(S_prev.F, 2));
    currentRot = rotationMatrixToVector(worldOrientation);
    currentRot_normalized = currentRot / norm(currentRot);
    for i=1:length(canBeAdded)
        rot = reshape(S_prev.T(:, i), [3, 4]);
        rot = rotationMatrixToVector(rot(1:3, 1:3));
        cosAngle = dot(currentRot_normalized, rot)/norm(rot);
        if (cosAngle < bearingAngleCosThreshold)
            canBeAdded(i) = 1;
        end
    end
    new_points = candidate_points(canBeAdded, :);
    new_T = S_prev.T(:, canBeAdded);
    new_F = S_prev.F(:, canBeAdded);
    new_landmarks = zeros(3, size(new_points, 1));
    camMatrix2 = cameraMatrix(cameraParams, worldOrientation, worldLocation);
    for i=1:size(new_points, 1)
       T = reshape(new_T(:, i), [3, 4]);
       camMatrix1 = cameraMatrix(cameraParams, T(1:3, 1:3), T(:, 4));
       worldPoint = triangulate(new_F(:, i)', new_points(i, :), camMatrix1,camMatrix2);
       new_landmarks(:, i) = worldPoint';
    end
    S_i.X = [S_prev.X, new_landmarks]; % add new landmark
    key_points = [key_points; new_points];
    setPoints(keyPointTracker, key_points);% add new tracking point
    
    % update candidates
    candidate_points = candidate_points(~canBeAdded, :);
    S_i.F = S_prev.F(:, ~canBeAdded);
    S_i.T = S_prev.T(:, ~canBeAdded); % removed candidates that are already keypoints
    
    % add new candidates
    featurePoints = detectHarrisFeatures(I_i);
    featurePoints = featurePoints.location;
    zero_img = zeros(size(I_i));
    one_img = ones(size(I_i));
    occupied_points = [key_points; candidate_points];
    occupied_points = floor(occupied_points + 0.5);
    ind = sub2ind(size(I_i), occupied_points(:, 2), occupied_points(:, 1));
    one_img(ind) = 0;
    ind = sub2ind(size(I_i), featurePoints(:, 2), featurePoints(:, 1));
    zero_img(ind) = 1;
    featurePoints = zero_img .* one_img;
    [row, col] = find(featurePoints);
    featurePoints = [col, row];
    candidate_points = [candidate_points; featurePoints];
    setPoints(candidateTracker, candidate_points);% add new tracking candidate
    numOfFeature = size(featurePoints, 1);
    S_i.F = [S_i.F, featurePoints'];
    S_i.T = [S_i.T, repmat(T_i(:), [1, numOfFeature])];
end

