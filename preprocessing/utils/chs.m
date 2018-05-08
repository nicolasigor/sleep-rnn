function A=chs(in,val1)

% in=[0 1 0 1 1 0  0 1];
% val1=1;
in=[in 0];
index=1:length(in);pos=zeros(length(index),1);

pos=0*index;
for ik=1:length(val1)
old_pos=pos;  
pos=pos+(in==val1(ik));

end
 pos=index(logical(pos));%!!2

count=1;A=[];flag=0;
 for i=1:length(pos)-1
if pos(i+1)-pos(i)==1 && flag==0
A(count,1)=pos(i);
flag=1;
elseif pos(i+1)-pos(i)>10 && flag==1
A(count,2)=pos(i);
flag=0;
count=count+1;
else
A(count,2)=pos(i);
end
 end
 u=size(A,1);
 if size(A,1)>0
 if A(u,2)==0
A(u,2)=A(u,1);
 end
 
  if A(u,1)==0
A(u,1)=A(u,2);
  end
 end