function transfer_learnng()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
net=load('alexnet-caffe');
net=vl_simplenn_tidy(net);
fcn_shape=net.layers{end-1}.size(3);
img_norm_meta=net.meta.normalization;
if ~exist('imdb.mat','file')
    image_database(img_norm_meta.imageSize(1),net.meta.normalization.averageImage);
end
data=load('imdb.mat');
net.layers=net.layers(1:end-2);
feat=vl_simplenn(net,data.images.data);
a=squeeze(feat(end).x).';
save('AlexNet2012','a');
end

