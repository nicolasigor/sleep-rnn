function [aux,record,header]=OPENregist(aux,FileName,Path)
expPath=[Path  strrep(FileName,'.rec','') '/Sleep Spindles/' ];
expFileName=['SS1_' strrep(FileName,'.rec','') '.txt'];

hypPath=[Path  strrep(FileName,'.rec','') '/Sleep States/' ];
hypFileName=['States_' strrep(FileName,'.rec','') '.txt'];

deltaPath=[Path  strrep(FileName,'.rec','') '/Delta Waves/' ];
deltaFileName=['DW1_' strrep(FileName,'.rec','') '.txt'];

remPath=[Path  strrep(FileName,'.rec','') '/REM/' ];
remFileName=['REM_' strrep(FileName,'.rec','') '.txt'];

if isequal(FileName,0)
return
else   
%Carga Datos registo completo
[record,header]=OPENedf(FileName,1,2,Path,Path);
aux.Global_Final=str2num(header.data_records);
aux.Fs=header.samples(1);
aux.FileName=FileName;
aux.Path=Path;
aux.expFileName=expFileName;
aux.expPath=expPath;
aux.hypfilename=hypFileName;
aux.hyppath=hypPath;
aux.deltafilename=deltaFileName;
aux.deltapath=deltaPath;
aux.remfilename=remFileName;
aux.rempath=remPath;
aux.header=header;



%Carga Registo 
% end
end
end
