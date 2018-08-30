function values=ttsigmoid(n,cores)
M=size(n,1);
values=zeros(M,1);
for j=1:M
    core=cores{1};
    value=permute(core(:,n(j,1),:),[1,3,2]);
    for i=2:6
        core=cores{i};
        value=value*permute(core(:,n(j,i),:),[1,3,2]);
    end
    values(j)=value;
end
%Relu
%if value<0
%    value=0;
%end
%sigmoid
values=1./(1+exp(-values));
