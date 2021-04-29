from PyPDF2 import PdfFileMerger
import os


folders = [
	'BECA011405_SQ2_cycle1',
	'BECA011405_SQ2_cycle2',
	'BECA011405_SQ2_cycle3',
	'BECA011405_SQ2_cycle4',
	'BECA011405_SQ2_cycle5',
	'BECA011405_SQ2_cycle6',
	'BECA011405_SQ2_cycle7',
	'BECA011405_SQ2_cycle8'
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
