function Q = StructureMeasure(prediction,GT)


% Check input
if (~isa(prediction,'double'))
    error('The prediction should be double type...');
end
if ((max(prediction(:))>1) || min(prediction(:))<0)
    error('The prediction should be in the range of [0 1]...');
end
if (~islogical(GT))
    error('GT should be logical type...');
end

y = mean2(GT);

if (y==0)% if the GT is completely black
    x = mean2(prediction);
    Q = 1.0 - x; %only calculate the area of intersection
elseif(y==1)%if the GT is completely white
    x = mean2(prediction);
    Q = x; %only calcualte the area of intersection
else
    alpha = 0.5;
    Q = alpha*S_object(prediction,GT)+(1-alpha)*S_region(prediction,GT);
    if (Q<0)
      Q=0;
    end
end

end
