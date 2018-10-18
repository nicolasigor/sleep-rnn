import pyedflib

regs = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13',
        '14', '15', '16', '17', '18', '19']
print(len(regs))
for reg in regs:
    path_eeg_file = "sleep_data/ssdata_mass/register/01-02-00"+reg+" PSG.edf"

    channel = 13  # C3 in 13, F3 in 22
    file = pyedflib.EdfReader(path_eeg_file)
    # signal = file.readSignal(channel)
    # print("Signal shape: ", signal.shape)

    # fs_old = file.getSampleFrequency(channel)
    # print("Supposed Fs: ", fs_old)

    # signal_duration = file.file_duration
    # print("Supposed signal duration")

    fs_direct = file.samplefrequency(channel)
    print("Direct fs for reg", reg, ":", fs_direct)

    file._close()
    del file

print("According to EDF file:", str(512/1.99987228))
