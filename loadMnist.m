function [xTrainImages,tTrain, xTestImages, tTest] = loadMnist()
train_data_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz';
train_label_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz';
test_data_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz';
test_label_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz';
%filename = 'mnist_train_data';
%websave(filename,train_data_url);
gunzip(train_data_url);
gunzip(train_label_url);
gunzip(test_data_url);
gunzip(test_label_url);

 % first we have to open the binary file



trainImgs = fopen('train-images-idx3-ubyte','r','b');
MagicNumber=fread(trainImgs,1,'int32');
nImages= fread(trainImgs,1,'int32');% Read the number of images
xTrainImages = cell(1,nImages);

fseek(trainImgs,16,'bof');
for k = 1:60000
    img= fread(trainImgs,28*28,'uchar');
    img2=zeros(28,28);
    for i=1:28
        img2(i,:)=img((i-1)*28+1:i*28);
    end
    %img2=img2(5:24,5:24);
    img2 = img2 ./255;
    xTrainImages{k} = img2;
end
figure,imshow(img2);


trainLabels = fopen('train-labels-idx1-ubyte','r','b');
MagicNumber=fread(trainLabels,1,'int32');
nLabels= fread(trainLabels,1,'int32');
fseek(trainLabels,8,'bof');
tTrainNormal = zeros(nLabels,1);
for i = 1:nLabels
    tTrainNormal(i) = fread(trainLabels,1,'uchar');
end

%Convert to onehot
tTrain = zeros(10,nLabels);
for i = 1:nLabels
    lbl = tTrainNormal(i);
    tTrain(lbl+1,i) = 1;
end

%Read Test Data

testImgs = fopen('t10k-images-idx3-ubyte','r','b');
MagicNumber=fread(testImgs,1,'int32');
nImages= fread(testImgs,1,'int32');% Read the number of images
xTestImages = cell(1,nImages);

fseek(testImgs,16,'bof');
for k = 1:nImages
    img= fread(testImgs,28*28,'uchar');
    img2=zeros(28,28);
    for i=1:28
        img2(i,:)=img((i-1)*28+1:i*28);
    end
    %img2=img2(5:24,5:24);
    img2 = img2 ./255;
    xTestImages{k} = img2;
end
figure,imshow(img2);

testLabels = fopen('t10k-labels-idx1-ubyte','r','b');
MagicNumber=fread(testLabels,1,'int32');
nLabels= fread(testLabels,1,'int32');
fseek(testLabels,8,'bof');
tTestNormal = zeros(nLabels,1);
for i = 1:nLabels
    tTestNormal(i) = fread(testLabels,1,'uchar');
end

%Convert to onehot
tTest = zeros(10,nLabels);
for i = 1:nLabels
    lbl = tTestNormal(i);
    tTest(lbl+1,i) = 1;
end

end
