function y = nanfilt(data,b,a,Order)
% column of data: time
% each column has common nans (detect only first column)

ind = find(any(isnan(data),2)) ;
y = NaN(size(data)) ;

if ~isempty(ind)  
    int(:,2) = ind-1 ;
    int(1,1) = 1 ;
    int(length(ind)+1,2) = size(data,1) ;
    int(2:end,1) = ind(1:end)+1 ;
    rep = length(ind)+1 ;
    
else ind = 1 ;
    int = [1 size(data,1)] ;
    rep = 1 ;
end

for t = 1:rep
    if int(t,2)-int(t,1)+1 > 3*Order
        y(int(t,1):int(t,2),:) = filtfilt(b,a,data(int(t,1):int(t,2),:)) ;
    end
end




