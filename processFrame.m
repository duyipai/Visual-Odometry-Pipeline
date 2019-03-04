function [S_i, T_i] = processFrame(I_i, S_prev, cameraParams) % remember to use spacial coordinate
    global bearingAngleCosThreshold;
    rng(0);
    
    % first extract current pose
    global keyPointTracker candidateTracker;
    K = (cameraParams.IntrinsicMatrix)';
    [key_points,validity] = keyPointTracker(I_i);
    key_points = key_points(validity, :);
    S_prev.X = S_prev.X(:, validity);
    [worldOrientation,worldLocation, validity] = estimateWorldCameraPose(key_points, single(S_prev.X'), cameraParams);
%     if (size(key_points, 1) > 500)  
%         key_points = key_points(validity, :);
%         S_prev.X = S_prev.X(:, validity);
%     end
    T_i = [worldOrientation, worldLocation'];
    
    % then seek new tracking points from candidates
    [candidate_points, validity] = candidateTracker(I_i);
    candidate_points = candidate_points(validity, :);
    S_prev.F = S_prev.F(:, validity);
    S_prev.T = S_prev.T(:, validity);
    canBeAdded = false([1, size(S_prev.F, 2)]);
    cosAngle = zeros(size(canBeAdded));
    for i=1:length(canBeAdded)
        this_T = reshape(S_prev.T(:, i), [3, 4]);
        this_location = this_T(1:3, 4);
        this_orientation = this_T(1:3, 1:3);
        this_init_point = K\[S_prev.F(:, i); 1];
        vector_init = this_orientation * this_init_point + this_location;
        this_current_point = K\[(candidate_points(i, :))'; 1];
        vector_current = worldOrientation * this_current_point + worldLocation';
        cosAngle(i) = dot(vector_init, vector_current)/norm(vector_init)/norm(vector_current);
        if (cosAngle(i) < bearingAngleCosThreshold)
            canBeAdded(i) = true;
        end
    end
    if (sum(canBeAdded) > 5)
        new_points = candidate_points(canBeAdded, :);
        new_T = S_prev.T(:, canBeAdded);
        new_F = S_prev.F(:, canBeAdded);% but why detect feature gives non integer points?
        one_img = ones(size(I_i));
        occupied_points = key_points;
        occupied_points = floor(occupied_points + 0.5);
        ind = sub2ind(size(I_i), occupied_points(:, 2), occupied_points(:, 1));
        one_img(ind) = 0;
        i = 1;
        while(i <= size(new_points, 1))
           if (one_img(floor(new_points(i, 2)+0.5), floor(new_points(i, 1)+0.5)) == 0)
               new_points(i, :) = [];
               new_T(:, i) = [];
               new_F(:, i) = [];
           else
               i = i + 1;
           end
        end
        
        new_landmarks = zeros(3, size(new_points, 1));
        camMatrix2 = cameraMatrix(cameraParams, worldOrientation', -worldLocation*worldOrientation');
        for i=1:size(new_points, 1)
           T = reshape(new_T(:, i), [3, 4]);
           camMatrix1 = cameraMatrix(cameraParams, (T(1:3, 1:3))', -T(:, 4)'*(T(1:3, 1:3))');
           worldPoint = triangulate(new_F(:, i)', new_points(i, :), camMatrix1, camMatrix2);
           new_landmarks(:, i) = worldPoint';
        end
%         S_i.X = [S_prev.X, new_landmarks]; % add new landmark
%         key_points = [key_points; new_points];
        
        S_i.X = new_landmarks; % add new landmark
        key_points = new_points;
    else
        S_i.X = S_prev.X;
    end
    
    setPoints(keyPointTracker, key_points);% add new tracking point
    
    % update candidates
    %if(sum(canBeAdded) ~= size(canBeAdded, 2))
    candidate_points = candidate_points(~canBeAdded, :);
    S_i.F = S_prev.F(:, ~canBeAdded);
    S_i.T = S_prev.T(:, ~canBeAdded); % removed candidates that are already keypoints 
    %end
    
    % add new candidates
    featurePoints = detectHarrisFeatures(I_i);
    featurePoints = featurePoints.Location; % but why detect feature gives non integer points?
    one_img = ones(size(I_i));
    occupied_points = candidate_points;
    occupied_points = floor(occupied_points + 0.5);
    ind = sub2ind(size(I_i), occupied_points(:, 2), occupied_points(:, 1));
    one_img(ind) = 0;
    i = 1;
    while(i <= size(featurePoints, 1))
       if (one_img(floor(featurePoints(i, 2)+0.5), floor(featurePoints(i, 1)+0.5)) == 0)
           featurePoints(i, :) = [];
       else
           i = i + 1;
       end
    end
    candidate_points = [candidate_points; featurePoints];
%     release(candidateTracker);
    setPoints(candidateTracker, candidate_points);% add new tracking candidate
    numOfFeature = size(featurePoints, 1);
    S_i.F = [S_i.F, featurePoints'];
    S_i.T = [S_i.T, repmat(T_i(:), [1, numOfFeature])];
end

