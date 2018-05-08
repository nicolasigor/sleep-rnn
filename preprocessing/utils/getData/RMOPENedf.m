function [record,header]=OPENedf(nombrearchivo,data_start,data_end,pathname,startpath)
%tic;
%--------------------------------------------------------------------------
% falta activar pagina de inicio y final
%
%
%
%
%
%
%--------------------------------------------------------------------------
cd(pathname);

fid=fopen(nombrearchivo,'r');

fseek(fid,0,-1);
fseek(fid,0,'eof');
total_bytes=ftell(fid);%si el numero de data records es desconocido
fseek(fid,0,-1); %vuelvo al inicio
header.version=fgets(fid,8); 
header.patient=fgets(fid,80);
header.local_recording=fgets(fid,80);
header.startdate=fgets(fid,8);
header.starttime=fgets(fid,8);
header.bytes_header=fgets(fid,8);    
n_bytes_header=eval(header.bytes_header);  %cantidad de bytes en el header
reserved1=fgets(fid,44);
header.data_records=fgets(fid,8);
n_data_records=eval(header.data_records);%cantidad de data records
header.duration_data=fgets(fid,8);
duration_data=eval(header.duration_data);
header.signal=fgets(fid,4);
nsig=eval(header.signal); %cantidad de señales
for i=1:nsig
    header.label(i)={fgets(fid,16)};
end
for i=1:nsig
    header.transducer(i)={fgets(fid,80)};
end
for i=1:nsig
    header.physical_dimension(i)={fgets(fid,8)};
end
%lectura de datos importantes
%minimos fisicos
Min_fi=zeros(nsig,1);
for i=1:nsig
    header.Min_fi(i)=str2double(fgets(fid,8));
end
%maximos fiscos
Max_fi=zeros(nsig,1);
for i=1:nsig
    header.Max_fi(i)=str2double(fgets(fid,8));
end
%minimos digitales
Min_di=zeros(nsig,1);
for i=1:nsig
    header.Min_di(i)=str2num(fgets(fid,8));
end
%maximos digitales
Max_di=zeros(nsig,1);
for i=1:nsig
    header.Max_di(i)=str2num(fgets(fid,8));
end
for i=1:nsig
    header.prefiltering(i)={fgets(fid,80)};
end
%frecuencias de muestro por canal
for i=1:nsig
    header.samples(i)=str2num(fgets(fid,8));
end
for i=1:nsig
    reserved={fgets(fid,32)};
end
%la cantidad de data records puede ser desconocida (-1)
num_data_records=sum(header.samples);
bytes_data_records2=2*num_data_records;
n_data_records2=(total_bytes-n_bytes_header)/bytes_data_records2;
% inicio del data records y comienzo de lectura de datos
% me dirijo a la pagina de inicio
pos_actual=ftell(fid);
fseek(fid,2*num_data_records*data_start,0);
ftell(fid);
inicio=zeros(nsig,1);
final=zeros(nsig,1);
inicio(1)=1;
for i=2:nsig
    inicio(i)=inicio(i-1)+header.samples(i-1);
end
final(1)=header.samples(1);
for i=2:nsig
    final(i)=final(i-1)+header.samples(i);
end
for i=1:data_end-data_start+1
    for j=1:nsig
        [mat,cout]=fread(fid,header.samples(j),'int16');
        canal(inicio(j):final(j),i)=mat;
    end
end
for i=1:nsig
    record{i}=(reshape(canal(inicio(i):final(i),:),header.samples(i)*(data_end-data_start+1),1));
end

fclose(fid);
%cd(directorio);
%t=toc;
cd(startpath);
end