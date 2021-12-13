function d = dist2( X, Y )
  if nargin==0
    d=[];
    return;
  end
  if nargin==1
    Y=X;
  end
  if(size(X,2)>1)
      d=repmat(sum(X'.^2)',1,size(Y,1));
      d=d+repmat(sum(Y'.^2),size(X,1),1);
  else
      d=repmat(X.^2,1,size(Y,1));
      d=d+repmat(Y'.^2,size(X,1),1);
  end
  d=abs(d-2*X*Y');
  if size(X,2)>10
      d=d./size(X,2);
  end
end%function