clear;
d=6; 
n=[10,10,8,8,4,2]'; 
r=[1,4,4,4,4,4,1]';
load cores;
fun = @(ind,c) ttsigmoid(ind,c);
tic;
tt1=amen_cross_vector(n,fun,1e-2,cores);
%tt1=dmrg_cross(d,n,fun,1e-2,cores);
toc;
A=full(tt1);
tic;
tt2=tt_tensor;
tt2.n=n;
tt2.d=d;
tt2.r=r;
tt2.ps=cumsum([1;n.*r(1:d).*r(2:d+1)]);
for i=1:d
    cr1=cores{i};
    cr(tt2.ps(i):(tt2.ps(i+1)-1))=cr1(:);
end
tt2.core=cr;
B=full(tt2);
B=1./(1+exp(-B));
toc;
error=norm(A(:)-B(:))/norm(B(:));      