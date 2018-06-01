function Experto=OPENexp(Experto,inicio,final,Fs,canal)
if size(Experto,1)~=0
b=final-inicio;
a=Fs*inicio;
b=(b*Fs)+Fs;
c=4;
Experto(:,[1 2])=Experto(:,[1 2])-a;
Experto=Experto(Experto(:,1)>=-c*Fs,: );
Experto=Experto(Experto(:,2)<=b+(c+Fs),: );
Experto=Experto(Experto(:,6)==canal,:);
Experto=Experto((Experto(:,2)-Experto(:,1))/Fs>0.5,:);
Experto=Experto( Experto(:,2)>0 ,:);
Experto=Experto(Experto(:,1)<b ,:);
% % ->cortar espacios inicio final
n=1:size(Experto,1);%husos al principio epoca
n_neg_E=n(Experto(:,1)<=0);
Experto(n_neg_E,1)=1;
n_sup_E=n(Experto(:,2)>=b);
Experto(n_sup_E,2)=b;

%%


else
    Experto=[];
end
cd(pwd)
end
