% Step 1: Clear the workspace and command window
clear all;
close all;
clc;

% Path to the dataset folder
datasetPath = 'at&tface-database'; % Make sure this folder contains the images

% Number of classes (people) and images per class
numClasses = 40; % Total number of people
imagesPerClass = 10; % Number of images per person

% Image size (all images will be resized to this)
imageSize = [112, 92];

dataMatrix = []; % This will store all the image data
labels = []; % This will store the class labels

for class = 1:numClasses
    % Folder path for the current class (e.g., s1, s2, ..., s40)
    folderPath = fullfile(datasetPath, sprintf('s%d', class));
    
    for imgIdx = 1:imagesPerClass
        % Get the full path of the image
        imgPath = fullfile(folderPath, sprintf('%d.pgm', imgIdx));
        
        % Read the image
        img = imread(imgPath);
        
        % Resize the image to the specified size
        img = imresize(img, imageSize);
        
        % Flatten the image into a row vector and add it to dataMatrix
        dataMatrix = [dataMatrix; double(img(:))']; 
        
        % Add the class label to the labels array
        labels = [labels; class]; 
    end
end

% normalize the image data (scale pixel values between 0 and 1)
dataMatrix = dataMatrix / 255;

% plit data into training and testing sets
trainRatio = 0.7; % 70% of the data for training
totalImages = numClasses * imagesPerClass;
numTrainImages = round(trainRatio * totalImages);

% randomize the order of images
randomIndices = randperm(totalImages);

% Determine the indices for the training data
trainIndices = randomIndices(1:numTrainImages);

% Extract training data based on the random indices
trainData = dataMatrix(trainIndices, :);

%train the data
trainLabels = labels(trainIndices);

% Determine the indices for the testing data
testIndices = randomIndices(numTrainImages+1:end);

% Extract testing data based on the random indices
testData = dataMatrix(testIndices, :);

% Extract testing labels based on the random indices
testLabels = labels(testIndices);

disp('Data split complete.');

% Step 6: Apply PCA to reduce dimensionality
[coeff, ~, ~, ~, explained] = pca(trainData);

% Determine the number of components needed to explain 85% of the variance
cumulativeVariance = cumsum(explained);
numComponents = find(cumulativeVariance >= 85, 1);

% Reduce training and testing data dimensions
trainFeatures = trainData * coeff(:, 1:numComponents);
testFeatures = testData * coeff(:, 1:numComponents);

 classifier = fitcdiscr(trainFeatures, trainLabels);

predictedLabels = predict(classifier, testFeatures);

% Calculate the accuracy
accuracy = sum(predictedLabels == testLabels) / numel(testLabels) * 100;
disp(['Classification Accuracy: ', num2str(accuracy), '%']);

% Visualization
% PCA Eigenface Visualization
randomIndex = randi(size(trainData, 1)); % Random index from the training data
originalFace = trainData(randomIndex, :); % Random face

% Reconstruct the face  
reconstructedFace = mean(trainData, 1) + (originalFace - mean(trainData, 1)) * coeff(:, 1:numComponents) * coeff(:, 1:numComponents)';

% Plot the original face and PCA-transformed face
figure;
subplot(1, 2, 1);
imshow(reshape(originalFace, imageSize), []);
title('Original Face');

subplot(1, 2, 2);
imshow(reshape(reconstructedFace, imageSize), []);
title('PCA-Transformed Face');
sgtitle('Original vs. PCA-Transformed Face');


% PCA Visualization (Before and After Dimensionality Reduction)
figure;
subplot(1, 2, 1);
scatter(trainData(:, 1), trainData(:, 2), 10, trainLabels, 'filled');
title('Original Feature Space');
xlabel('Feature 1');
ylabel('Feature 2');

subplot(1, 2, 2);
scatter(trainFeatures(:, 1), trainFeatures(:, 2), 10, trainLabels, 'filled');
title('PCA-Reduced Feature Space');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
sgtitle('Before and After PCA');

% 3. Confusion Matrix Heatmap
confMatrix = confusionmat(testLabels, predictedLabels);

figure;
heatmap(confMatrix, 'Title', 'Confusion Matrix', ...
    'XLabel', 'Predicted Class', 'YLabel', 'True Class', ...
    'ColorbarVisible', 'on');

uniqueClasses = unique(testLabels); % Find unique classes in the test labels
numUniqueClasses = length(uniqueClasses); % Number of unique classes

% Reinitialize precision and recall based on unique classes
precision = zeros(numUniqueClasses, 1);
recall = zeros(numUniqueClasses, 1);

% Calculate precision and recall for each unique class
for classIdx = 1:numUniqueClasses
    currentClass = uniqueClasses(classIdx); % Map to the actual class label
    TP = confMatrix(classIdx, classIdx); % True Positives
    FP = sum(confMatrix(:, classIdx)) - TP; % False Positives
    FN = sum(confMatrix(classIdx, :)) - TP;
    
    precision(classIdx) = TP / (TP + FP + eps); % Precision
    recall(classIdx) = TP / (TP + FN + eps); % Recall
end

% Average precision and recall across all unique classes
avgPrecision = mean(precision);
avgRecall = mean(recall);

% Print Metrics
disp(['Accuracy: ', num2str(accuracy), '%']);
disp(['Average Precision: ', num2str(avgPrecision)]);
disp(['Average Recall: ', num2str(avgRecall)]);

