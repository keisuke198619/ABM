function y = chi2test(tab,type)
% type 1: compute expCounts using the averaged value
% type 2: compute expCounts using Poisson distribution
% see https://jp.mathworks.com/help/stats/chi2gof.html

bins = 0:length(tab(:))-1 ;
n = sum(tab(:)) ;
[lx,ly] = size(tab) ;
if type == 1
    nr = sum(tab,2) ;
    mr = mean(tab,1)./sum(mean(tab,1)) ;
    expCounts = repmat(nr,1,ly).*repmat(mr,lx,1) ;
elseif type == 2 % Poisson distribution
    pd = fitdist(bins','Poisson','Frequency',tab(:));
    expCounts = n * pdf(pd,bins);
end

[y.h,y.p,y.st] = chi2gof(bins,'ctrs',bins,'frequency',tab(:),...
    'expected',expCounts(:),'alpha',0.05) ;
y.CramerV = sqrt(y.st.chi2stat/(n*min(lx-1,ly-1))) ;