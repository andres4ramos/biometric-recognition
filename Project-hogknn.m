clear all;
close all;
clc;

% Path to the dataset folder
datasetPath = 'at&tface-database';
numClasses = 40; % Total number of people
imagesPerClass = 10; % Number of images per person

imageSize = [112, 92]; % Resizing the image

hogFeatureMatrix = []; % Store all the HOG feature data
labels = []; % Store class labels

for class = 1:numClasses
    folderPath = [datasetPath, '/s', num2str(class)];
    
    for imgIdx = 1:imagesPerClass
        imgName = [num2str(imgIdx), '.pgm'];
        imgPath = [folderPath, '/', imgName];
        
        img = imread(imgPath);
        
        img = imresize(img, imageSize);
        
        % Extract HOG features 
        hogFeatures = extractHOGFeatures(img);
        
       % Append HOG features to the feature matrix
    if  isempty(hogFeatureMatrix)
        hogFeatureMatrix = hogFeatures;
    else
        hogFeatureMatrix = [hogFeatureMatrix; hogFeatures]; % Add new features as a row
end

% Append the class label to the label list
if isempty(labels)
    labels = class; 
else
    labels = [labels; class]; 
end
    end
end

% Split the data into training and testing sets
trainRatio = 0.70; % 70% of the data for training
totalImages = size(hogFeatureMatrix, 1);

% Calculate how many images should be used for training (70% of the total)
numTrainImages = round(trainRatio * totalImages);

randomIndices = randperm(totalImages);

% Create training and testing sets
trainFeatures = hogFeatureMatrix(randomIndices(1:numTrainImages), :);
trainLabels = labels(randomIndices(1:numTrainImages));
testFeatures = hogFeatureMatrix(randomIndices(numTrainImages+1:end), :);
testLabels = labels(randomIndices(numTrainImages+1:end));

% Training
knnModel = fitcknn(trainFeatures, trainLabels, 'NumNeighbors',3);

%testing
predictedLabels = predict(knnModel, testFeatures);

% Calculate confusion matrix 
confMatrix = confusionmat(testLabels, predictedLabels);

% Calculate overall accuracy
correctPredictions = sum(diag(confMatrix));
totalPredictions = sum(confMatrix(:));
accuracyK3 = (correctPredictions / totalPredictions) * 100;

% Initialize precision and recall
numClasses = size(confMatrix, 1); % Number of classes
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);

% calculate precision and recall for each class
for classIdx = 1:numClasses
    TP = confMatrix(classIdx, classIdx); % True Positives
    FP = sum(confMatrix(:, classIdx)) - TP; % False Positives
    FN = sum(confMatrix(classIdx, :)) - TP; % False Negatives
    
    precision(classIdx) = TP / (TP + FP + eps); % Precision
    recall(classIdx) = TP / (TP + FN + eps); % Recall
end

%  average precision and recall across all classes
avgPrecision = mean(precision);
avgRecall = mean(recall);

% Display metrics
disp('Metrics for k = 3:');
fprintf('Accuracy: %.2f%%\n', accuracyK3);
fprintf('Average Precision: %.2f\n', avgPrecision);
fprintf('Average Recall: %.2f\n', avgRecall);

%  Plotting Histograms  
selectedClass = 1; % Choose a sample class
selectedImageIdx = 1; % Index of the image in the class
imgName = [num2str(selectedImageIdx), '.pgm'];
imgPath = [datasetPath, '/s', num2str(selectedClass), '/', imgName];

% Read the original image
originalImg = imread(imgPath);

% Resize the image (as in your pipeline)
resizedImg = imresize(originalImg, imageSize);

equalizedImg = histeq(resizedImg);

figure;
subplot(1, 3, 1);
imhist(originalImg);
title('Original Image Histogram');

subplot(1, 3, 2);
imhist(resizedImg);
title('Resized Image Histogram');

subplot(1, 3, 3);
imhist(equalizedImg);
title('Equalized Image Histogram');

% Define range of k values
kValues = 1:10; % Number of neighbors to test
accuracies = zeros(1, length(kValues)); % To store accuracy for each k

% test and train  the model
for i = 1:length(kValues)
    knnModel = fitcknn(trainFeatures, trainLabels, 'NumNeighbors', kValues(i));
    
    predictedLabels = predict(knnModel, testFeatures);
    
    correctPredictions = sum(predictedLabels == testLabels); 
    accuracies(i) = (correctPredictions / length(testLabels)) * 100;
end

% Plot the accuracy for each k value
figure;
plot(kValues, accuracies, '-o'); 
xlabel('Number of Neighbors (k)');  
ylabel('Accuracy (%)'); % Label y-axis
title('k-NN Accuracy vs. Number of Neighbors'); 
grid on; % Show grid for better readability


% make a confusion Matrix 
figure; % Create a new figure
heatmap(confMatrix);  
title('Confusion Matrix');  
xlabel('Predicted Class');  
ylabel('True Class');  
colorbar;  
