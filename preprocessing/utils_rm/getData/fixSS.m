function new_spindles = fixSS( spindles )

duration=spindles(:,2)-spindles(:,1);
spindles=spindles(and(duration>=100,duration<=600),:);% duration e [0.5,3.0]seg

n=length(spindles);
distSS=zeros(n,1);
distSS(1)=600;
for d=2:n
    distSS(d)=spindles(d,1)-spindles(d-1,2);
end

new_spindles=zeros(n,2);
idxNew=1;
new_spindles(idxNew,:)=spindles(1,:);
for i=2:n
    if(distSS(i)>100)
        idxNew=idxNew+1;
        new_spindles(idxNew,:)=spindles(i,:);
    else
        new_spindles(idxNew,2)=spindles(i,2);
    end
end
new_spindles=new_spindles(1:idxNew,:);
end

