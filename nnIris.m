close all; clear; clc;
load fisheriris.mat;

dataset = meas;
speciesNumber= grp2idx(species);
dataset = [dataset speciesNumber];

Sdata_partition = cvpartition(speciesNumber, 'HoldOut', 0.4);
trainDataIndex = training(Sdata_partition);
testDataIndex = test(Sdata_partition);

X_trainData = meas(trainDataIndex, :); %Features - Traing data X_Train 
T_trainData = speciesNumber(trainDataIndex); %Lables - Training target data

X_testData= meas(testDataIndex, :); %Features - Testing data
T_testData = speciesNumber(testDataIndex); %Lables  - Testing target data

%Use a Feedforward Neural Network

noNeurons = 5; %number of neurons
totalAccuracy = 0 ; %Total Accuracy 

for i = 1:4
    NnTrainFunction = 'trainlm'; %Neural Network Training function
    hiddenLayers_size = noNeurons; %size of the hidden layer to number of neurons
    nNet = feedforwardnet(hiddenLayers_size,NnTrainFunction); %Define feedforwardnet Neural Network 
    nNet = train(nNet,X_trainData',T_trainData');
    
    for j = 1:10
        y = nNet(X_testData');
        y=round(y);      
        total=0;
        
        for k = 1:size(T_testData,1)
            if y(k)==T_testData(k)
                total = total +1;
            end
        end   

        accuracyValue = (total/size(T_testData,1))*10;
        totalAccuracy = totalAccuracy + accuracyValue;
    end

    %view(nNet);
    accuracyOutput = [' The average  performance accuracy for ',num2str(noNeurons),' hidden layer = ', num2str(totalAccuracy),'%'];
    disp(accuracyOutput)
    noNeurons = noNeurons+5; %to increment the value by 5
    totalAccuracy=0;
end









