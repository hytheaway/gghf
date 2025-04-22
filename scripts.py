# much of this is sourced from andrea genovese
# please find him here: https://andreagenovese.com/
# and his wonderful tutorial about hrtf processing here: https://www.youtube.com/watch?v=a4mpK_2koR4

# also please keep in mind that this isn't supposed to be "efficient" or "clean" or "lightweight".
# this is meant to be the most brute force way to do all my hrtf processing in one file with one interface. 

import numpy as np
import matplotlib.pyplot as plt
import sys, glob
import soundfile as sf # <- read audio
import sofa # <- read SOFA HRTFs
import librosa # <- resample function
from scipy import signal # <- fast convolution function
from IPython.display import Audio # <- Audio listening (in notebook)
import tkinter as tk
from tkinter import filedialog
import pygame

root = tk.Tk()
root.geometry('800x1000')
root.title('HRTF Functions')

titleLabel = tk.Label(root, text="\nGarrett's Great\nHRTF Functions\n")
titleLabel.grid(row=0, column=1)

selectHRTFFileButton = tk.Button(root, text='Select HRTF File', command=lambda:selectHRTFFile())
selectHRTFFileLabel = tk.Label(root, text='HRTF file:\n', wraplength=120)
selectHRTFFileButton.grid(row=3, column=0)
selectHRTFFileLabel.grid(row=4, column=0)

selectSourceFileButton = tk.Button(root, text='Select source file', command=lambda:selectSourceFile())
selectSourceFileLabel = tk.Label(root, text='Source file:\n', wraplength=120)
selectSourceFileButton.grid(row=3, column=2)
selectSourceFileLabel.grid(row=4, column=2)

getHRTFFileDataButton = tk.Button(root, text='Get HRTF File Data', state='disabled', command=lambda:getHRTFFileData())
getHRTFFileDataButton.grid(row=5, column=0)

getSourceFileDataButton = tk.Button(root, text='Get Source File Data', state='disabled', command=lambda:getSourceFileData())
getSourceFileDataButton.grid(row=5, column=2)

timeDomainVisualHRTFButton = tk.Button(root, text='HRTF Time Domain Visualization', state='disabled', command=lambda:timeDomainVisualHRTF())
timeDomainVisualHRTFButton.grid(row=6, column=0)

stereoToMonoButton = tk.Button(root, text='Source File Stereo -> Mono', state='disabled', command=lambda:stereoToMono())
stereoToMonoButton.grid(row=6, column=2)

freqDomainVisualHRTFButton = tk.Button(root, text='HRTF Frequency Domain Visualization', state='disabled', command=lambda:freqDomainVisualHRTF())
freqDomainVisualHRTFButton.grid(row=7, column=0)

resampleButton = tk.Button(root, text='Resample', state='disabled', command=lambda:fs_resample(sig_mono, fs_s, HRIR, fs_H))
resampleButton.grid(row=8, column=1)

timeDomainConvolveButton = tk.Button(root, text='Time Domain Convolve', state='disabled', command=lambda:timeDomainConvolve())
timeDomainConvolveButton.grid(row=9, column=1)

exportConvolvedButton = tk.Button(root, text='Export Convolved', state='disabled', command=lambda:exportConvolved())
exportConvolvedButton.grid(row=10, column=1)

# ---

sectionalLabel = tk.Label(root, text='\n--------------------------------------------------')
sectionalLabel.grid(row=11, column=1)

sofaLabel = tk.Label(root, text="Garrett's Great\nSOFA Functions\n")
sofaLabel.grid(row=12, column=1)

selectSOFAFileButton = tk.Button(root, text='Select SOFA File', command=lambda:selectSOFAFile())
selectSOFAFileLabel = tk.Label(root, text='SOFA file:\n', wraplength=120)
selectSOFAFileButton.grid(row=13, column=1)
selectSOFAFileLabel.grid(row=14, column=1)

getSOFAFileMetadataButton = tk.Button(root, text='Get SOFA File Metadata', state='disabled', command=lambda:getSOFAFileMetadata())
getSOFAFileMetadataButton.grid(row=15, column=0)

getSOFAFileDimensionsButton = tk.Button(root, text='Get SOFA File Dimensions', state='disabled', command=lambda:getSOFAFileDimensions())
getSOFAFileDimensionsButton.grid(row=15, column=2)

sofaMeasurementTextBox = tk.Text(root, state='disabled', height=1, width=5)
sofaMeasurementTextBox.grid(row=16, column=0)

sofaMeasurementLabel = tk.Label(root, text='Measurement Index\n(default: 0)\n')
sofaMeasurementLabel.grid(row=17, column=0)

sofaEmitterTextBox = tk.Text(root, state='disabled', height=1, width=5)
sofaEmitterTextBox.grid(row=16, column=2)

sofaEmitterLabel = tk.Label(root, text='Emitter\n(default: 1)\n')
sofaEmitterLabel.grid(row=17, column=2)

frequencyXLimTextBox = tk.Text(root, state='disabled', height=1, width=15)
frequencyXLimTextBox.grid(row=18, column=0)

frequencyXLimLabel = tk.Label(root, text='Frequency Range (Hz)\n[start, end]')
frequencyXLimLabel.grid(row=19, column=0)

magnitudeYLimTextBox = tk.Text(root, state='disabled', height=1, width=15)
magnitudeYLimTextBox.grid(row=18, column=2)

magnitudeYLimLabel = tk.Label(root, text='Magnitude (dB)\n[start, end]')
magnitudeYLimLabel.grid(row=19, column=2)

sofaViewButton = tk.Button(root, text='View SOFA File', state='disabled', command=lambda:displaySOFAFile(frequencyXLimTextBox.get(1.0, 'end-1c'), magnitudeYLimTextBox.get(1.0, 'end-1c'), sofaMeasurementTextBox.get(1.0, 'end-1c'), sofaEmitterTextBox.get(1.0, 'end-1c')))
sofaViewButton.grid(row=20, column=1)

azimuthTextBox = tk.Text(root, state='disabled', height=1, width=5)
azimuthTextBox.grid(row=21, column=0)

azimuthLabel = tk.Label(root, text='Desired azimuth (in deg)')
azimuthLabel.grid(row=22, column=0)

elevationTextBox = tk.Text(root, state='disabled', height=1, width=5)
elevationTextBox.grid(row=21, column=2)

elevationLabel = tk.Label(root, text='Desired elevation (in deg)')
elevationLabel.grid(row=22, column=2)

sofaDisplayButton = tk.Button(root, text='Render Source with SOFA file', state='disabled', command=lambda:manualSOFADisplay(azimuthTextBox.get(1.0, 'end-1c'), elevationTextBox.get(1.0, 'end-1c'), source_file))
sofaDisplayButton.grid(row=22, column=1)


# ---

quitButton = tk.Button(root, text='Quit', command=quit)
quitButton.grid(row=40, column=1)

# ---

def playAudio(audio_file):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

def selectHRTFFile():
    global hrtf_file
    global hrtf_file_print
    global HRIR
    global fs_H
    hrtf_file = filedialog.askopenfilename()
    if hrtf_file:
        hrtf_file_print = hrtf_file.split('/')
        selectHRTFFileLabel.config(text='HRTF file:\n' + hrtf_file_print[len(hrtf_file_print)-1] + '\n')
        getHRTFFileDataButton.config(state='active')
        timeDomainVisualHRTFButton.config(state='active')
        freqDomainVisualHRTFButton.config(state='active')
        resampleButton.config(state='disabled')
        [HRIR,fs_H] = sf.read(hrtf_file)
    else:
        return

def selectSourceFile():
    global source_file 
    global source_file_print
    global sig
    global fs_s
    source_file = filedialog.askopenfilename()
    if source_file:
        source_file_print = source_file.split('/')
        selectSourceFileLabel.config(text='Source file:\n' + source_file_print[len(source_file_print)-1] + '\n')
        getSourceFileDataButton.config(state='active')
        stereoToMonoButton.config(state='active')
        resampleButton.config(state='disabled')
        [sig,fs_s] = sf.read(source_file)
    else:
        return

def getHRTFFileData():
    newWindow = tk.Toplevel(root)
    newWindow.geometry('400x120')
    newWindow.title('HRTF File Data')
    windowTitleHRTFData = tk.Label(newWindow, text='\n' + hrtf_file_print[len(hrtf_file_print)-1] + '\n')
    windowTitleHRTFData.grid(row=0, column=1)
    hrtfSampleRateLabel = tk.Label(newWindow, text='Sample rate: ' + str(fs_H))
    hrtfSampleRateLabel.grid(row=1, column=0)
    hrtfDataDimLabel = tk.Label(newWindow, text='Data dimensions: ' + str(HRIR.shape))
    hrtfDataDimLabel.grid(row=1, column=2)
    hrtfPlayFileButton = tk.Button(newWindow, text='Play HRTF', command=lambda:playAudio(hrtf_file))
    hrtfPlayFileButton.grid(row=2, column=0)
    hrtfPauseFileButton = tk.Button(newWindow, text='Pause HRTF', command=lambda:pygame.mixer.music.pause())
    hrtfPauseFileButton.grid(row=2, column=2)

def getSourceFileData():
    newWindow = tk.Toplevel(root)
    newWindow.geometry('550x120')
    newWindow.title('Source File Data')
    windowTitleSourceData = tk.Label(newWindow, text='\n' + source_file_print[len(source_file_print)-1] + '\n')
    windowTitleSourceData.grid(row=0, column=1)
    sourceSampleRateLabel = tk.Label(newWindow, text='Sample rate: ' + str(fs_s))
    sourceSampleRateLabel.grid(row=1, column=0)
    sourceDataDimLabel = tk.Label(newWindow, text='Data dimensions: ' + str(sig.shape))
    sourceDataDimLabel.grid(row=1, column=2)
    sourcePlayFileButton = tk.Button(newWindow, text='Play Source File', command=lambda:playAudio(source_file))
    sourcePlayFileButton.grid(row=2, column=0)
    sourcePauseFileButton = tk.Button(newWindow, text='Pause Source File', command=lambda:pygame.mixer.music.pause())
    sourcePauseFileButton.grid(row=2, column=2)

def timeDomainVisualHRTF():
    plt.plot(HRIR[:,0]) # left channel data
    plt.plot(HRIR[:,1]) # right channel data
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('HRIR at angle: ' + hrtf_file_print[len(hrtf_file_print)-1])
    plt.legend(['Left','Right'])
    plt.show()

def freqDomainVisualHRTF():
    nfft = len(HRIR)*8
    HRTF = np.fft.fft(HRIR,n=nfft, axis=0)
    print(HRIR)
    print(HRTF)
    HRTF_mag = (2/nfft)*np.abs(HRTF[0:int(len(HRTF)/2)+1,:])
    HRTF_mag_dB = 20*np.log10(HRTF_mag)

    f_axis = np.linspace(0,fs_H/2,len(HRTF_mag_dB))
    plt.semilogx(f_axis, HRTF_mag_dB)
    plt.grid()
    plt.grid(which='minor', color="0.9")
    plt.title('HRTF at angle: ' + hrtf_file_print[len(hrtf_file_print)-1])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend(['Left','Right'])
    plt.show()

def stereoToMono():
    global sig_mono
    if len(sig.shape) > 1:
        if sig.shape[1] > 1:
            sig_mono = np.mean(sig, axis=1)
        else:
            sig_mono = sig
        newWindow = tk.Toplevel(root)
        newWindow.geometry('350x100')
        newWindow.title('Stereo -> Mono')
        windowTitleStereoToMonoLabel = tk.Label(newWindow, text='\n' + 'New source data dimensions: ' + str(sig_mono.shape) + '\n')
        windowTitleStereoToMonoLabel.grid(row=0, column=1)
    else:
        sig_mono = sig
        newWindow = tk.Toplevel(root)
        newWindow.geometry('350x100')
        newWindow.title('Stereo -> Mono')
        windowTitleStereoToMonoLabel = tk.Label(newWindow, text='\n' + 'Source file already mono.' + '\n')
        windowTitleStereoToMonoLabel.grid(row=0, column=1)
    resampleButton.config(state='active')
    
def fs_resample(s1, f1, s2, f2):
    if f1 != f2:
        if f2 < f1:
            s2 = librosa.core.resample(s2.transpose(), f2, f1)
            s2 = s2.transpose()
        else:
            s1 = librosa.core.resample(s1.transpose(), f1, f2)
            s1.transpose()
    fmax = max([f1, f2])
    f1 = fmax
    f2 = fmax
    global resampleWindow
    global windowTitleResampleLabel
    resampleWindow = tk.Toplevel(root)
    resampleWindow.geometry('250x100')
    resampleWindow.title('Resample')
    windowTitleResampleLabel = tk.Label(resampleWindow, text='\n' + 'Resampled at: ' + str(fmax) + 'Hz' + '\n')
    windowTitleResampleLabel.grid(row=0, column=1)
    resampleSignalSourceDimensionsLabel = tk.Label(resampleWindow, text='Signal/source dimensions: ' + str(sig_mono.shape))
    resampleSignalSourceDimensionsLabel.grid(row=1, column=1)
    resampleHRIRDimensionsLabel = tk.Label(resampleWindow, text='HRIR Dimensions: ' + str(HRIR.shape))
    resampleHRIRDimensionsLabel.grid(row=2, column=1)
    timeDomainConvolveButton.config(state='active')
    return s1, f1, s2, f2

def timeDomainConvolve():
    global Bin_Mix
    s_L = np.convolve(sig_mono,HRIR[:,0])
    s_R = np.convolve(sig_mono,HRIR[:,1])

    Bin_Mix = np.vstack([s_L,s_R]).transpose()
    newWindow = tk.Toplevel(root)
    newWindow.geometry('250x100')
    newWindow.title('Time Domain Convolve')
    timeDomainConvolveWindowLabel = tk.Label(newWindow, text='\n' + 'Data dimensions: ' + str(Bin_Mix.shape) + '\n')
    timeDomainConvolveWindowLabel.grid(row=0, column=1)
    exportConvolvedButton.config(state='active')

# freq domain convolution goes here

def exportConvolved():
    source_file_export_name = source_file_print[(len(source_file_print)-1)]
    hrtf_file_export_name = hrtf_file_print[(len(hrtf_file_print)-1)]

    sf.write(str(source_file_export_name[:-4]) + '-' + str(hrtf_file_export_name[:-4]) + '-export.wav', Bin_Mix, fs_s)

def selectSOFAFile():
    global sofa_file
    global sofa_file_print
    sofa_file = filedialog.askopenfilename()
    if sofa_file:
        sofa_file_print = sofa_file.split('/')
        selectSOFAFileLabel.config(text='SOFA file:\n' + sofa_file_print[len(sofa_file_print)-1] + '\n')
        getSOFAFileMetadataButton.config(state='active')
        getSOFAFileDimensionsButton.config(state='active')
        sofaDisplayButton.config(state='active')
        sofaViewButton.config(state='active')
        sofaMeasurementTextBox.config(state='normal')
        sofaEmitterTextBox.config(state='normal')
        azimuthTextBox.config(state='normal')
        elevationTextBox.config(state='normal')
        frequencyXLimTextBox.config(state='normal')
        magnitudeYLimTextBox.config(state='normal')
    else:
        return

def getSOFAFileMetadata():
    newWindow = tk.Toplevel(root)
    newWindow.geometry('400x400')
    newWindow.title('SOFA File Metadata')
    v = tk.Scrollbar(newWindow, orient='vertical')
    v.pack(side='right', fill='y')
    myString = ''
    for attr in sofa.Database.open(sofa_file).Metadata.list_attributes():
        myString = myString + ("{0}: {1}".format(attr, sofa.Database.open(sofa_file).Metadata.get_attribute(attr))) + '\n'
    windowSOFAMetadataLabel = tk.Text(newWindow, width=100, height=100, wrap='word', yscrollcommand=v.set)
    windowSOFAMetadataLabel.insert('end', str(myString))
    windowSOFAMetadataLabel.pack()
    # for attr in sofa.Database.open(sofa_file).Data.dump():
    #     myString2 = myString2 + ("{0}: {1}".format(attr, sofa.Database.open(sofa_file).Data.dump(attr))) + '\n'
    # windowSOFADataLabel = tk.Text(newWindow, width=100, height=100, wrap='word', yscrollcommand=v.set)
    # windowSOFADataLabel.insert('end', str(myString2))
    # windowSOFADataLabel.pack()

    v.config(command=windowSOFAMetadataLabel.yview)

def getSOFAFileDimensions():
    newWindow = tk.Toplevel(root)
    newWindow.geometry('600x400')
    newWindow.title('SOFA File Dimensions')
    v = tk.Scrollbar(newWindow, orient='vertical')
    v.pack(side='right', fill='y')
    definitionsLabel = tk.Label(newWindow, text='C = Size of coordinate dimension (always three).\n\nI = Single dimension (always one).\n\nM = Number of measurements.\n\nR = Number of receivers or SH coefficients (depending on ReceiverPosition_Type).\n\nE = Number of emitters or SH coefficients (depending on EmitterPosition_Type).\n\nN = Number of samples, frequencies, SOS coefficients (depending on self.GLOBAL_DataType).')
    definitionsLabel.pack()
    myString = ''
    for dimen in sofa.Database.open(sofa_file).Dimensions.list_dimensions():
        myString = myString + ("{0}: {1}".format(dimen, sofa.Database.open(sofa_file).Dimensions.get_dimension(dimen))) + '\n'
    windowSOFADimensionsLabel = tk.Text(newWindow, width=10, height=10, wrap='word', yscrollcommand=v.set)
    windowSOFADimensionsLabel.insert('end', str(myString))
    windowSOFADimensionsLabel.pack()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def exportSOFAConvolved(source_file_name, angle_label, elev_label, audioContent, samplerate=48000):
    source_file_export_name = source_file_name[(len(source_file_name)-1)]
    sf.write(str(source_file_export_name[:-4]) + '-' + str(angle_label) + '_' + str(elev_label) + '-export.wav', audioContent, samplerate)

def plot_coordinates(coords, title):
    x0 = coords
    n0 = coords
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    q = ax.quiver(x0[:, 0], x0[:, 1], x0[:, 2], 
                  n0[:, 0], n0[:, 1], n0[:, 2], length=0.1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    return q

def displaySOFAFile(xlim, ylim, measurement=0, emitter=1):
    if measurement == '':
        measurement = 0
    if emitter == '':
        emitter = 1
    SOFA_HRTF = sofa.Database.open(sofa_file)

    # plot source coordinates
    source_positions = SOFA_HRTF.Source.Position.get_values(system='cartesian')
    plot_coordinates(source_positions, 'Source positions')

    measurement = measurement
    emitter = emitter
    legend = []

    xlim = str(xlim)
    for char in xlim:
        if char == '[' or char == ']' or char == ' ':
            xlim = xlim.replace(char, '')
    for i in range(0, len(xlim)):
        if xlim[i] == ',':
            xlim_start = xlim[0:i]
            xlim_end = xlim[i+1:]

    ylim = str(ylim)
    for char in ylim:
        if char == '[' or char == ']' or char == ' ':
            ylim = ylim.replace(char, '')
    for i in range(0, len(ylim)):
        if ylim[i] == ',':
            ylim_start = ylim[0:i]
            ylim_end = ylim[i+1:]

    t = np.arange(0, SOFA_HRTF.Dimensions.N)*SOFA_HRTF.Data.SamplingRate.get_values(indices={"M":measurement})

    plt.figure(figsize=(15, 5))
    for receiver in np.arange(SOFA_HRTF.Dimensions.R):
        plt.plot(t, SOFA_HRTF.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))
        legend.append('Receiver {0}'.format(receiver))
    plt.title('HRIR at M={0} for emitter {1}'.format(measurement, emitter))
    plt.legend(legend)
    plt.xlabel('$t$ in s')
    plt.ylabel(r'$h(t)$')
    plt.grid()

    plt.show()

    # not so sure this is the best way to do this but it's probably fine
    nfft = len(SOFA_HRTF.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))*8
    HRTF = np.fft.fft(SOFA_HRTF.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}),n=nfft, axis=0)
    print(HRTF)
    HRTF_mag = (2/nfft)*np.abs(HRTF[0:int(len(HRTF)/2)+1])
    HRTF_mag_dB = 20*np.log10(HRTF_mag)

    f_axis = np.linspace(0,(SOFA_HRTF.Data.SamplingRate.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))/2,len(HRTF_mag_dB))
    plt.semilogx(f_axis, HRTF_mag_dB)
    ax = plt.gca()
    ax.set_xlim([int(xlim_start), int(xlim_end)])
    ax.set_ylim([int(ylim_start), int(ylim_end)])
    plt.grid()
    plt.grid(which='minor', color="0.9")
    plt.title('HRTF at M={0} for emitter {1}'.format(measurement, emitter))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend(['Left','Right'])
    plt.show()

    SOFA_HRTF.close()
    plt.close()

def manualSOFADisplay(angle, elev, source_file, target_fs=48000):
    global Stereo3D
    if not angle or not elev or not source_file:
        newWindow = tk.Toplevel(root)
        newWindow.geometry('300x100')
        newWindow.title('Error')

        errorLabel = tk.Label(newWindow, text='Missing azimuth, elevation, or source file')
        errorLabel.pack()
        return -1

    # init
    SOFA_HRTF = sofa.Database.open(sofa_file)
    sofa_fs_H = SOFA_HRTF.Data.SamplingRate.get_values()[0]
    sofa_positions = SOFA_HRTF.Source.Position.get_values(system='spherical')
    SOFA_H = np.zeros([SOFA_HRTF.Dimensions.N,2])
    Stereo3D = np.zeros([SOFA_HRTF.Dimensions.N,2])

    newWindow = tk.Toplevel(root)
    newWindow.geometry('550x150')
    newWindow.title('SOFA Rendering')
    windowHRTFInfoLabel = tk.Label(newWindow, text='Using HRTF set: ' + sofa_file_print[len(sofa_file_print)-1])
    windowHRTFInfoLabel.grid(row=1, column=1)
    try:
        windowSourceDistanceLabel = tk.Label(newWindow, text='Source distance is: ' + str(sofa_positions[0,2]) + 'meters\n')
        windowSourceDistanceLabel.grid(row=2, column=1)
    except:
        windowSourceNoDistanceLabel = tk.Label(newWindow, text='No distance information available\n')
        windowSourceNoDistanceLabel.grid(row=2, column=1)

    angle = int(angle)
    elev = int(elev)

    # database specific format adjustments
    global angle_label
    global elev_label
    angle_label = angle
    elev_label = elev
    angle = 360 - angle
    if angle == 360:
        angle = 0

    # retrieve hrtf data for angle
    [az, az_idx] = find_nearest(sofa_positions[:,0], angle)
    subpositions = sofa_positions[np.where(sofa_positions[:,0]==az)]
    [el, sub_idx] = find_nearest(subpositions[:,1],elev)
    SOFA_H[:,0] = SOFA_HRTF.Data.IR.get_values(indices={"M":az_idx+sub_idx, "R":0, "E":0})
    SOFA_H[:,1] = SOFA_HRTF.Data.IR.get_values(indices={"M":az_idx+sub_idx, "R":1, "E":0})
    if sofa_fs_H != target_fs:
        SOFA_H = librosa.core.resample(SOFA_H.transpose(),sofa_fs_H, target_fs).transpose()

    # pick a source (already picked)
    [source_x, fs_x] = sf.read(source_file)
    if len(source_x.shape)>1:
        if source_x.shape[1] > 1:
            source_x = np.mean(source_x, axis=1)
    if fs_x != target_fs:
        source_x = librosa.core.resample(source_x.transpose(), fs_x, target_fs).transpose()

    rend_L = signal.fftconvolve(source_x,SOFA_H[:,0])
    rend_R = signal.fftconvolve(source_x,SOFA_H[:,1])
    M = np.max([np.abs(rend_L), np.abs(rend_R)])
    if len(Stereo3D) < len(rend_L):
        diff = len(rend_L) - len(Stereo3D)
        Stereo3D = np.append(Stereo3D, np.zeros([diff,2]),0)
    Stereo3D[0:len(rend_L),0] += (rend_L/M)
    Stereo3D[0:len(rend_R),1] += (rend_R/M)

    windowSOFAInfoLabel = tk.Label(newWindow, text='Source: ' + str(source_file_print[len(source_file_print)-1]) + ' rendered at azimuth ' + str(angle_label) + ' and elevation ' + str(elev))
    windowSOFAInfoLabel.grid(row=3, column=1)

    exportSOFAConvolved(str(source_file_print[len(source_file_print)-1]), angle_label, elev_label, Stereo3D, int(sofa_fs_H))

    return Stereo3D

root.mainloop()