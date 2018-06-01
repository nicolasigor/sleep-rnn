function [ aux, param, out ] = RMgetSSdata()
%aux: parametros auxiliares para la lectura de datos
%param: parametros para leer datos
%out: resultados


aux.subject='ADGU101504';
aux.foldername='REGISTROS';
aux.fileRecord=[aux.foldername '/' aux.subject '.rec'];
aux.fileSleepSates=[aux.foldername '/' aux.subject '/Sleep States/States_' aux.subject '.txt'  ];
aux.fileSpleepSpindles=[aux.foldername '/' aux.subject '/Sleep Spindles/SS1_' aux.subject '.txt'  ];
aux.pathname=fileparts(which(aux.fileRecord));
aux.startpath=pwd;

param.channel=1;
param.Twin=30;
param.Fs=200;
param.context=5;
param.cycle=3;%N2
param.frecRange=[10,16];
%
data=struct('pageNum',{},'record',{},'firstSampleNum',{},'lastSampleNum',{},...
    'spindlesExpert',{},'spindlesExpert_Mlen',{},'spindlesDetected',{},...
    'TP',{},'FP',{},'FN',{},'FPreal',{},'sobreEST',{},...
    'TPidx',{}, 'FPidx',{},'FNidx',{},'sobIdx',{},...
    'filteredRecord',{},'butter',{},'imf',{},...
    'logicalSpindlesExpert',{},'logicalSpindlesExpert_Mlen',{},...
    'logicalSpindlesDetected',{},...
    'sampleNumbers',{},'Tstart',{},'Tend',{});
%load
load(aux.fileSpleepSpindles);
load(aux.fileSleepSates);

param.numPagesInRecord=length(States_ADGU101504);


%spindles

marksSpindlesCH1=SS1_ADGU101504((SS1_ADGU101504(:,6)==param.channel),:);%canal 1
marksSpindlesCH1=marksSpindlesCH1((marksSpindlesCH1(:,5)~=0),:);%los 0 son invalidos
%%NEW
marksSpindlesCH1=marksSpindlesCH1(:,[1 2 5]);
n=length(marksSpindlesCH1);
spindlesCH1=zeros(size(marksSpindlesCH1));
write=1;
spindlesCH1(write,:)=marksSpindlesCH1(1,:);
for read=2:n
    %write=last one written
    inter=~isempty(intersect((floor(spindlesCH1(write,1)):floor(spindlesCH1(write,2))),(floor(marksSpindlesCH1(read,1)):floor(marksSpindlesCH1(read,2)))));
    validity=marksSpindlesCH1(read,3);
    if (validity==1)
        if(inter==0)
            write=write+1;
            spindlesCH1(write,:)=marksSpindlesCH1(read,:);
        elseif(inter==1)
            ignore=1;
        end
    elseif (validity==2)
        if(inter==0)
            write=write+1;
            spindlesCH1(write,:)=marksSpindlesCH1(read,:);
        elseif(inter==1)
            spindlesCH1(write,:)=marksSpindlesCH1(read,:);
        end
    end
end
spindlesCH1=spindlesCH1(1:write,1:2);
SSe=fixSS(spindlesCH1);
out.spindlesExpert=SSe;
%{
n=length(marksSpindlesCH1);
spindlesCH1=zeros(size(marksSpindlesCH1));

%corregir spindles
write=0;
for read=1:n
    if(marksSpindlesCH1(read,5)==1)
        write=write+1;
        spindlesCH1(write,:)=marksSpindlesCH1(read,:);
    elseif(marksSpindlesCH1(read,5)==2)
        spindlesCH1(write,:)=marksSpindlesCH1(read,:);
    end
end
spindlesCH1=spindlesCH1(1:write,1:2);

SSe=fixSS(spindlesCH1);
out.spindlesExpert=SSe;
%}

%read data pages
%filter
[B,A]=butter(3,[param.frecRange(1)/(param.Fs/2) param.frecRange(2)/(param.Fs/2)] ,'bandpass');

for page=1:length(States_ADGU101504)
    if(States_ADGU101504(page,8)==param.cycle)
        n=length(data)+1;
        %
        data(n).pageNum=page;
        data(n).Tstart=((page-1)*param.Twin)-param.context;
        data(n).Tend=((page*param.Twin)-1)+param.context;
        data(n).firstSampleNum=data(n).Tstart*param.Fs;
        data(n).lastSampleNum=(data(n).Tend+1)*param.Fs;
        data(n).sampleNumbers=data(n).firstSampleNum:1:data(n).lastSampleNum;
            data(n).sampleNumbers=data(n).sampleNumbers(1:data(n).lastSampleNum-data(n).firstSampleNum);
        spindlesExpert=out.spindlesExpert;
            spindlesExpert=spindlesExpert(spindlesExpert(:,1)>=data(n).firstSampleNum,:);
            spindlesExpert=spindlesExpert(spindlesExpert(:,2)<data(n).lastSampleNum,:);
        data(n).spindlesExpert=spindlesExpert;
        %
        [record,~]=RMOPENedf(aux.fileRecord,data(n).Tstart,data(n).Tend,aux.pathname,aux.startpath);
        data(n).record=record{1,param.channel};
        imf=emd( data(n).record,'maxmodes',5);
        data(n).butter=filtfilt(B,A,data(n).record);
        N_maxEMD=5;
        data(n).imf=imf(1,:);
        %data(n).filteredRecord=data(n).imf;
        data(n).filteredRecord=data(n).butter;
    end
end
out.data=data;

for p=1:length(out.data)
    %disp(p)
    if(~isempty(out.data(p).spindlesExpert))
        logic=zeros(size(out.data(p).record));
        [R,~]=size(length(out.data(p).spindlesExpert));
        for r=1:R
            %disp([out.data(p).spindlesExpert(r,1)-out.data(p).firstSampleNum,out.data(p).spindlesExpert(r,2)-out.data(p).firstSampleNum])
            logic(out.data(p).spindlesExpert(r,1)-out.data(p).firstSampleNum:out.data(p).spindlesExpert(r,2)-out.data(p).firstSampleNum)=r;
        end
        out.data(p).logicalSpindlesExpert=logic;
    end
end

end

