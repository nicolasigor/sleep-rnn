from PyPDF2 import PdfFileMerger
import os

part_folders = os.listdir('.')
part_folders = [f for f in part_folders if os.path.isdir(f)]
part_folders.sort()

for single_part in part_folders:
	pdfs = os.listdir(single_part)
	pdfs = [f for f in pdfs if '.pdf' in f]
	pdfs.sort()
	pdfs = [os.path.join(single_part, f) for f in pdfs]
	merger = PdfFileMerger()
	for pdf in pdfs:
		merger.append(pdf)
	merger.write("%s.pdf" % single_part)
	merger.close()
