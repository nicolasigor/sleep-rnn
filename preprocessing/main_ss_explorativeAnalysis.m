fs = eegData.params.fs;
interval = (167002-5000:167002+5000);
time = tabData(interval,1)/fs;
plot(time,tabData(interval,2),time,tabData(interval,3),time,100*tabData(interval,5))

% Ver las marcas raras del final en el archivo original de marcas, 1,2,3?