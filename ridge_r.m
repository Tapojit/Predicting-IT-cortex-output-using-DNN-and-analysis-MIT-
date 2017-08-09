function theta = ridge_r( Y,X,alpha )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% x1=gpuArray(X(:,1));
% % por1=arrayfun(@(x2) arrayfun(@(x) mtimes(gpuArray(X(:,x2)).',gpuArray(X(:,x))),1:size(X,2),'UniformOutput',false),1:size(X,2),'UniformOutput',false);
% x=gpuArray(X);
% y=gpuArray(Y);
x_f=size(X,2);
% x_f=x_f(2);
G=(alpha^2)*eye(x_f);
G(1,1)=0;
% prod=mtimes(x(:,1).',x(:,1));
% pt1=X.'*X;
% portion=mod(size(X,2),3000);
% part1=inv(mult(X.',X,portion)+G);
% part2=mult_2(X.',Y,portion);

theta=mtimes(inv(mtimes(X.',X)+G),mtimes(X.',Y));


end

function prod=mult(a,b,last_por)
prod=zeros(size(a,1));
block_count=(size(a,1)-last_por)/3000;
for f=1:block_count
    
    for i=1:block_count
        prod(((1+(f-1)*3000):3000*f),((1+(i-1)*3000):3000*i))=gather(gpuArray(a(((1+(f-1)*3000):3000*f),:))*gpuArray(b(:,((1+(i-1)*3000):3000*i))));
    end
    
    prod(((1+(f-1)*3000):3000*f),(block_count*3000)+1:size(a,1))=gather(gpuArray(a(((1+(f-1)*3000):3000*f),:))*gpuArray(b(:,(end-last_por+1):end)));
end

for i=1:block_count
    prod(((block_count*3000)+1:size(a,1)),((1+(i-1)*3000):3000*i))=gather(gpuArray(a(((block_count*3000)+1:size(a,1)),:))*gpuArray(b(:,((1+(i-1)*3000):3000*i))));
end
prod(((block_count*3000)+1:size(a,1)),(block_count*3000)+1:size(a,1))=gather(gpuArray(a(((block_count*3000)+1:size(a,1)),:))*gpuArray(b(:,(end-last_por+1):end)));

end
function prod=mult_2(a,b,last_por)
prod=zeros(size(a,1),1);
block_count=(size(a,1)-last_por)/3000;
for f=1:block_count
    
    prod(((1+(f-1)*3000):3000*f),1)=gather(gpuArray(a(((1+(f-1)*3000):3000*f),:))*gpuArray(b(:,1)));
end

prod(((block_count*3000)+1:size(a,1)),1)=gather(gpuArray(a(((block_count*3000)+1:size(a,1)),:))*gpuArray(b(:,1)));

end



