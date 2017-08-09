
function imdb = image_database (shape,mean_image)
%returns:
% i) binary labels of data
% ii) true-labels of data
% iii) shuffled array of directories in given file


file_dir='PLoSCB2014_data_20141216/PLoSCB2014_data_20141216';
meta=load([file_dir '/NeuralData_IT_multiunits']);
meta=meta.meta;
imdb.meta.description={};
imdb.images.labels={};
imdb.images.data=single(zeros(shape,shape,3,size(meta,1)));
disp('Starting data matrix creation: ');
for a=1:size(meta,1)
    disp([num2str(a) ' of ' num2str(size(meta,1))]);
    img_path=meta(a,(1:51));
    imdb.images.data(:,:,:,a)=img_resizer(file_dir,img_path);
    [imdb.meta.description{end+1} imdb.images.labels{end+1}]=labeler(meta(a,:));
end
disp('Data matrix creation done');
imdb.meta.sets={'train','validation','test'};
imdb.meta.classes={'Animals','Cars','Chairs','Faces','Fruits','Planes','Tables'};
imdb.images.set=[ones(1,1568) 3*ones(1,392)];
imdb.images.data_mean=mean_image;
disp('Calculating rectified data: ');
imdb.images.data=bsxfun(@minus,imdb.images.data,mean_image);
disp('Rectified data calculation done: ');
save(fullfile(pwd,'imdb.mat'),'-struct','imdb');
end

function [desc, ind]=labeler(meta_str)
%labels indexes 1 or 2
if strcmp(meta_str(1,(53:59)),'Animals')
    desc='Animals';
    ind=1;
elseif strcmp(meta_str(1,(53:56)),'Cars')
    desc='Cars';
    ind=2;
elseif strcmp(meta_str(1,(53:58)),'Chairs')
    desc='Chairs';
    ind=3;
elseif strcmp(meta_str(1,(53:57)),'Faces')
    desc='Faces';
    ind=4;
elseif strcmp(meta_str(1,(53:58)),'Fruits')
    desc='Fruits';
    ind=5;
elseif strcmp(meta_str(1,(53:58)),'Planes')
    desc='Planes';
    ind=6;
elseif strcmp(meta_str(1,(53:58)),'Tables')
    desc='Tables';
    ind=7;
end

end
function img=img_resizer(path,img_path)
img_dir=[path '/' img_path];
img=single(imread(img_dir));
img=img((1+14:256-15),(1+14:256-15),:);

end





