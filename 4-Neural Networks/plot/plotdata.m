function plotdata(d,s)
% plotdata(data,format);
    if nargin<2
        s = 'xb';
        end
    if size(d,2)==1
        plot(d(:,1),d(:,1),s);
    elseif size(d,2)==2
        plot(d(:,1),d(:,2),s);
    else
        plot3(d(:,1),d(:,2),d(:,3),s);
    end
end