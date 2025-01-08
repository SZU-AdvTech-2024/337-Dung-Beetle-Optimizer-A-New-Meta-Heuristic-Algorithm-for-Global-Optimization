
% This function initialize the first population of search agents
function [x, new_lb, new_ub] = Initialization(N,dim,ub,lb)
Boundary= size(ub,2); % numnber of boundaries
new_lb = lb;
new_ub = ub;
% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary==1
    x=rand(N,dim).*(ub-lb)+lb;
    new_lb = lb*ones(1,dim);
    new_ub = ub*ones(1,dim);
end
% If each variable has a different lb and ub
if Boundary>1
    for i=1:dim
        ubi=ub(i);
        lbi=lb(i);
        x(:,i)=rand(N,1).*(ubi-lbi)+lbi;
    end
end