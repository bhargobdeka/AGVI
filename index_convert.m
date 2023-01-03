clear;clc;
n = 5;
m = triu(ones(n));
total = n*(n+1)*0.5;
index_diag = sub2ind(size(m),[1:1:n],[1:1:n]);
values = 1:1:total;
values_var = values(1:1:n);
values(1:1:n)=[];
for i  = 1:n-1
index_cov{i}  = sub2ind(size(m),[i*ones(1,n-i)],[i+1:1:n]);
values_cov{i} = values(1:size(index_cov{1,i},2));
values(1:1:size(index_cov{1,i},2))=[];
m(index_cov{1,i})=values_cov{1,i};
end
m(index_diag)=values_var;
ind=triu(true(size(m)));
index = m(ind)';

clear;clc;
n = 5;
total = n*(n+1)*0.5;
m = logical(triu(ones(n)));
A = zeros(n);
A(m)=[1:1:total];
A = A.';
diag_A = diag(A);
m1 = tril(true(size(A)),-1);
index=[diag_A;A(m1)]'



