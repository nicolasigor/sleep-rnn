from PyPDF2 import PdfFileMerger
import os


folders = [
	'BTOL083105_conflicts',
	'BRCA062405_conflicts',
	'CRCA020205_conflicts',
	'ESCI031905_conflicts',
	'BRLO041102_conflicts',
	'BTOL090105_conflicts',
	'BECA011405_conflicts',
	'ADGU101504_conflicts',
	'BECA011405_SQ2-SQ3'
]


for single_folder in folders:
	pdfs = os.listdir(single_folder)
	pdfs = [f for f in pdfs if '.pdf' in f]
	pdfs.sort()
	pdfs = [os.path.join(single_folder, f) for f in pdfs]
	merger = PdfFileMerger()
	for pdf in pdfs:
		merger.append(pdf)
	merger.write("%s.pdf" % single_folder)
	merger.close()
