function [net, Val_accuracy, d1, d2] =  DLTraining(training_data_path)

    %Importing Folder containing the labeled files. 
    %The function imageDatastore() generates a database of images 
    imds = imageDatastore(training_data_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    %countLabels=countEachLabel(imds);

    %Dividing out dataset into Training and Validation set. This should be
    %roughly in the ratio 70:30
    numTrainFiles = 700;
    [TrainingSet, ValidationSet, TestSet]= splitEachLabel(imds,0.7, 0.2, 0.1, 'randomize');
    %size(TestSet)
    %size(TrainingSet)
    %size(ValidationSet)
    
    %Input layer size is determined by the size of image. Therefore we extract
    %the size in x,y,z dimensions. Z-dimension specify the number of color
    %features in an image.
    image_size=readimage(imds,1);
    d1=size(image_size,1);
    d2=size(image_size,2);
    d3=size(image_size,3);

    %We now define the meta-structure of our Deep Learning Neural Network
    %General Architecture:
    %Conv_Layer->BatchNorm_Layer->ReLU_Layer->Pooling_Layer
    %For our network, we have added 5 Layers between Input and output layer
    % Input_Layer--L1--L2--L3--L4--L5--Output_Layer

    layers = [
        imageInputLayer([d1 d2 d3],'Name','InputLayer')

        convolution2dLayer(3,8,'Padding',1,'Name','conv_1')
        batchNormalizationLayer('Name','BN_1')
        reluLayer('Name','relu_1')

        maxPooling2dLayer(2,'Stride',2,'Name','max_pool_1')

        convolution2dLayer(3,16,'Padding',1,'Name','conv_2')
        batchNormalizationLayer('Name','BN_2')
        reluLayer('Name','relu_2')

        maxPooling2dLayer(2,'Stride',2,'Name','max_pool_2')

        convolution2dLayer(3,32,'Padding',1,'Name','conv_3')
        batchNormalizationLayer('Name','BN_3')
        reluLayer('Name','relu_3')

        maxPooling2dLayer(2,'Stride',2,'Name','max_pool_3')

        convolution2dLayer(3,32,'Padding',1,'Name','conv_4')
        batchNormalizationLayer('Name','BN_4')
        reluLayer('Name','relu_4')

        maxPooling2dLayer(2,'Stride',2,'Name','max_pool_4')


        convolution2dLayer(3,16,'Padding',1,'Name','conv_5')
        batchNormalizationLayer('Name','BN_5')
        reluLayer('Name','relu_5')

        fullyConnectedLayer(3,'Name','full_c')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classifyL')];
    
    lgraph = layerGraph(layers);
    figure
    plot(lgraph)
    %Options specify the hyperparameters needed to train our network
    options = trainingOptions('adam', ...
        'MaxEpochs',15, ...
        'ValidationData',ValidationSet, ...
        'ValidationFrequency',30, ...
        'LearnRateDropFactor',0.30,...
        'LearnRateDropPeriod',5,...
        'Verbose',false, ...
        'MiniBatchSize',64,...
        'Plots','training-progress');

    %Running the DNN to train on our dataset
    net = trainNetwork(TrainingSet,layers,options);
    %Calculating accuracy on validation set
    YPred = classify(net,ValidationSet);
    YValidation = ValidationSet.Labels;
    Val_accuracy = sum(YPred == YValidation)/numel(YValidation);

    %Running the DNN to calculate validation against TEST set
    TestPred = classify(net, TestSet);
    TestValidation = TestSet.Labels;
    Test_accuracy = sum(TestPred == TestValidation)/ numel(TestValidation);
    
    fprintf('Val acc: %i\n', Val_accuracy );
    fprintf('Test acc: %i\n', Test_accuracy );
    
end