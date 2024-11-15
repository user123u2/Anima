# with HackRF recordings using signal preambles and replay attack reconstruction
from helpers.features import get_features
from helpers.noise_segments import get_noise_segments
from helpers.baseband import RFSignalDownmix
from helpers.gnu_demodulate import RFdemodulate
from helpers.gnu_reconstruct import RFreconstruct
from helpers.augment_fir import augment_iqdata
import os
import glob
import numpy as np
import csv
from scipy.signal import spectrogram
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import random

directories = os.listdir("../recordings")
directories.remove('.DS_Store')
directories = sorted(directories, key=lambda x: int(x.split(' ')[0]))

def phase_detection(signal, threshold=0.96):
    phase = np.angle(signal)/(2 * np.pi)
    phase_diff = np.abs(np.diff(phase))

    change_points = np.where(phase_diff > threshold)[0] + 1
    if len(change_points) > 0:
        final_transients = [change_points[0]]
        x = len(final_transients)
        for i in range(1,len(change_points)-1):
            if (x%2 == 1) & (change_points[i+1] - change_points[i] >= 500000):
                final_transients.append(change_points[i])
                x+=1
            elif x%2 == 0:
                final_transients.append(change_points[i])
                x+=1
        final_transients.append(change_points[-1])
    else:
        final_transients = []

    return final_transients

current_directory = "../recordings"
subdirectory_name = "datasets/signals/"
fileNames = ["RTL-SDR","HackRF"]
fs = 2e6
fc_nominal = 433.92e6
m = 0
while abs(fc_nominal - m*fs) > fs // 2:
    m += 1
fc_aliased = abs(fc_nominal - m*fs)
print("Aliased carrier frequency: ", fc_aliased)

lower_cutoff_freq = 46e3
upper_cutoff_freq = 120e3
transition_bandwidth = 10e3

# datasets: legitimate, reconstructed, attack, augmented legitimate signals
with open(subdirectory_name + 'reconstructed.npy','wb') as f:
    pass
with open(subdirectory_name + 'legitimate.npy','wb') as f:
    pass
with open(subdirectory_name + 'attack1.npy','wb') as f:
    pass
with open(subdirectory_name + 'attack2.npy','wb') as f:
    pass
with open(subdirectory_name + 'augmented_legitimate.npy','wb') as f:
    pass

num_augmentations = 2
for directory in directories:
    for file in fileNames:
        if file == 'RTL-SDR':
            read_path = os.path.join(current_directory,directory, f'{file}*.complex')
        else:
            read_path = os.path.join(current_directory,directory, 'replay',f'{file}*')
        print('Read path:', glob.glob(read_path))
        print('File:', file)
        for path in glob.glob(read_path):
            iq_data = np.fromfile(path, dtype=np.complex64)
            threshold = 0.96
            change_points = phase_detection(iq_data,threshold)
            while ((len(change_points)!=10) or not all((change_points[i + 1] - change_points[i] >= 500000) for i in range(0, len(change_points) - 1, 2))) and threshold>=0.9:
                threshold-=0.01
                change_points = phase_detection(iq_data,threshold)
            if threshold<0.9:
                print("segmentation failed")
                continue
            check = all((change_points[i + 1] - change_points[i] >= 500000) for i in range(0, len(change_points) - 1, 2))
            if check:
                print("segmentation successful")
                print("threshold:",threshold,"change point indices:", len(change_points), change_points)

                for i in range(0,len(change_points) - 1, 2):
                    start_idx = change_points[i]
                    end_idx = change_points[i + 1]
                    print('Path:',path, "start_idx:", start_idx, "end_idx:", end_idx)
                    segment = iq_data[start_idx:end_idx]

                    tb = RFSignalDownmix(segment, fs, carrier_freq=82e3, lower_cutoff_freq=lower_cutoff_freq, upper_cutoff_freq=upper_cutoff_freq, transition_bandwidth=transition_bandwidth)
                    tb.run()
                    downmixed_signal = np.array(tb.downmix_sink.data())
                    filtered_baseband = np.array(tb.filtered_sink.data())

                    filtered_baseband = filtered_baseband / np.max(np.abs(filtered_baseband))

                    if 'gain' in path:
                        continuous_noise_segments, p_noise = get_noise_segments(filtered_baseband, file, atten_noise=False)
                    else:
                        continuous_noise_segments, p_noise = get_noise_segments(filtered_baseband, file)
                    preamble = filtered_baseband[0:continuous_noise_segments[0][0] - (continuous_noise_segments[1][0] - continuous_noise_segments[0][1])]
                    # preamble = filtered_baseband[0:continuous_noise_segments[0][0]]
                    preamble = preamble/np.max(np.abs(preamble))

                    # if file == 'RTL-SDR':
                        # preamble1 = filtered_baseband[continuous_noise_segments[0][1]:continuous_noise_segments[1][0]]
                        # preamble1 = preamble1/np.max(np.abs(preamble1))
                        # preamble2 = filtered_baseband[continuous_noise_segments[1][1]:]
                        # preamble2 = preamble2/np.max(np.abs(preamble2))
                        # preambles = [preamble,preamble1,preamble2]

                    f, t, Sxx = spectrogram(np.real(preamble), 2e6)

                    yf = fft(preamble)
                    freqs = fftfreq(len(preamble), 1/2e6)

                    yf = fft(preamble)
                    freqs = fftfreq(len(yf), d=1 / 2e6)
                    peak_frequency1 = np.abs(freqs[np.argmax(np.abs(yf))])
                    yf = np.abs(yf)
                    freqs = np.delete(np.abs(freqs), np.argmax(yf))
                    yf = np.delete(yf, np.argmax(yf))
                    peak_frequency2 = np.abs(freqs)[np.argmax(yf)]
                    while np.abs(peak_frequency1 - peak_frequency2) < 3000:
                        freqs = np.delete(np.abs(freqs), np.argmax(yf))
                        yf = np.delete(yf, np.argmax(yf))
                        peak_frequency2 = np.abs(freqs)[np.argmax(yf)]
                    peak_frequency3 = np.abs(freqs)[np.argmax(yf)]
                    while np.abs(peak_frequency2 - peak_frequency3) < 3000 or np.abs(peak_frequency1 - peak_frequency3) < 3000:
                        freqs = np.delete(np.abs(freqs), np.argmax(yf))
                        yf = np.delete(yf, np.argmax(yf))
                        peak_frequency3 = np.abs(freqs)[np.argmax(yf)]

                    lowcut = np.min([peak_frequency1, peak_frequency2,peak_frequency3])
                    highcut = np.max([peak_frequency1, peak_frequency2,peak_frequency3])
                    middle_freq = np.median([peak_frequency1, peak_frequency2,peak_frequency3])
                    print("peak frequencies:", lowcut, highcut, middle_freq)

                    if file == 'RTL-SDR':

                        # for reconsindex in range(7):
                        for reconsindex in range(2):
                            filename = f"HackRFReconstructed_{i//2}_{directory.split(' ')[0]}_{reconsindex}.complex"
                            print("Car key recording demodulation")

                            dc_offset = 0
                            sample_rate = 2e6
                            carrier_freq = 82e3
                            fsk_deviation= 30e3
                            baseband_lowcut = 46e3
                            baseband_uppercut = 120e3
                            transition_bandwidth = 10e3

                            dc_offset = np.abs(np.random.normal(0, 0.1))
                            carrier_freq_offset = np.random.normal(0,50)
                            sample_rate_offset = np.random.normal(0,40)
                            T_offset = np.random.uniform(0,60)
                            attenuation = np.random.uniform(0.5,0.8)
                            noise_level = np.random.uniform(0.05,0.2)
                            print("dc_offset:",dc_offset,"sample_rate_offset:",sample_rate_offset,"carrier_freq_offset:",carrier_freq_offset,"T_offset:",T_offset,"attenuation:",attenuation,"noise_level:",noise_level)

                            binary_data, p_noise = RFdemodulate(path,sample_rate=sample_rate,carrier_freq=carrier_freq,lower_cutoff_freq=baseband_lowcut,upper_cutoff_freq=baseband_uppercut,transition_bandwidth=transition_bandwidth,fc_deviation=fsk_deviation,threshold=0.95,T_offset=T_offset,carrier_freq_offset=carrier_freq_offset,sample_rate_offset=sample_rate_offset)
                            print("Signal Reconstruction")

                            reconstucted_preamble, upmixed_signal, noise = RFreconstruct(binary_data, sample_rate, carrier_freq=carrier_freq,fsk_deviation=fsk_deviation, dc_offset=dc_offset,attenuation=attenuation,noise_level=noise_level,sample_rate_offset=sample_rate_offset,carrier_freq_offset=carrier_freq_offset)
                            p_noise_reconstructed = np.sum(np.abs(noise)**2)/len(noise)
                            # with open(subdirectory_name + 'reconstructed7.npy','ab') as f:
                            #     np.save(f, reconstucted_preamble)
                            with open(subdirectory_name + 'reconstructed.npy','ab') as f:
                                np.save(f, reconstucted_preamble)
                    if path.split('/')[-1].startswith('HackRF1'):
                        filename = f"HackRF1_{i//2}_{directory.split(' ')[0]}.complex"
                        with open(subdirectory_name + 'attack1.npy','ab') as f:
                            np.save(f, preamble)
                    if path.split('/')[-1].startswith('HackRF2'):
                        filename = f"HackRF2_{i//2}_{directory.split(' ')[0]}.complex"
                        with open(subdirectory_name + 'attack2.npy','ab') as f:
                            np.save(f, preamble)
                    if path.split('/')[-1].startswith('RTL-SDR'):
                        filename = f"RTL-SDR_{i//2}_{directory.split(' ')[0]}.complex"
                        with open(subdirectory_name + 'legitimate.npy','ab') as f:
                            np.save(f, preamble)
                        # augment_legitimate = augment_iqdata(preamble, num_augmentations=5)
                        # with open(subdirectory_name + 'augmented_legitimate7.npy','ab') as f:
                        #     for augmented in augment_legitimate:
                        #         np.save(f, augmented)
                        augment_legitimate = augment_iqdata(preamble, num_augmentations=2)
                        with open(subdirectory_name + 'augmented_legitimate.npy','ab') as f:
                            for augmented in augment_legitimate:
                                np.save(f, augmented)
                            # for signal in preambles:
                            #     np.save(f, signal)
            else:
                print("segmentation failed")

print("done")