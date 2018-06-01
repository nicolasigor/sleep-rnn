function hipno=OPENhipno(FileName,Path,ncol)

if isequal(FileName,0)
return
else
Archivotxt=[Path FileName];
end
auxhipno=load(Archivotxt);

if size(auxhipno,1)~=0
hipno=auxhipno(:,ncol);
else
    hipno=[];
end

cd(pwd)
end
