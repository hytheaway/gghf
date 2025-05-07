# much of this is sourced from andrea genovese
# please find him here: https://andreagenovese.com/
# and his wonderful tutorial about hrtf processing here: https://www.youtube.com/watch?v=a4mpK_2koR4

# also please keep in mind that this isn't supposed to be "efficient" or "clean" or "lightweight".
# this is meant to be the most brute force way to do all my hrtf processing in one file with one interface. 

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys, glob
import soundfile as sf # <- read audio
import sofa # <- read SOFA HRTFs
import librosa # <- resample function
from scipy import signal # <- fast convolution function
from IPython.display import Audio # <- Audio listening (in notebook)
import tkinter as tk
from tkinter import filedialog
from tkinter import font
import pygame
# import pyinstaller
import pyi_splash
import PyQt6
from PyQt6 import QtWidgets
matplotlib.use('QtAgg')

pyi_splash.close()

def centered_window(window): #https://www.geeksforgeeks.org/how-to-center-a-window-on-the-screen-in-tkinter/
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

root = tk.Tk()
root.minsize(565, 890)
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
centered_window(root)
root.title("GGH&SF")
default_font = font.nametofont("TkDefaultFont")
parse_font_dict = default_font.actual()
# print(parse_font_dict)

rootFrame = tk.Frame(root, borderwidth=10, relief='flat')
rootFrame.grid(row=0, column=0)

titleLabel = tk.Label(rootFrame, text="Garrett's Great\nHRTF Functions\n", font=("TkDefaultFont", str(parse_font_dict['size'] + 2), "bold"))
titleLabel.grid(row=0, column=0, columnspan=3)

topSectionFrame = tk.Frame(rootFrame, borderwidth=10, relief='ridge')
topSectionFrame.grid(row=1, column=0, columnspan=3)

hrtfSourceSelectionFrame = tk.Frame(topSectionFrame, borderwidth=5, relief='flat')
hrtfSourceSelectionFrame.grid(row=0, column=0, columnspan=3)

hrtfFrame = tk.Frame(hrtfSourceSelectionFrame, borderwidth=10, relief='flat')
hrtfFrame.grid(row=0, column=0)

selectHRTFFileButton = tk.Button(hrtfFrame, text='Select HRTF File (.wav)', command=lambda:selectHRTFFile())
selectHRTFFileLabel = tk.Label(hrtfFrame, text='HRTF file:\n', wraplength=120)
selectHRTFFileButton.grid(row=0, column=0)
selectHRTFFileLabel.grid(row=1, column=0)
getHRTFFileDataButton = tk.Button(hrtfFrame, text='Get HRTF File Data', state='disabled', command=lambda:getHRTFFileData())
getHRTFFileDataButton.grid(row=3, column=0)
timeDomainVisualHRTFButton = tk.Button(hrtfFrame, text='HRTF Time Domain Visualization', state='disabled', command=lambda:timeDomainVisualHRTF())
timeDomainVisualHRTFButton.grid(row=4, column=0)
freqDomainVisualHRTFButton = tk.Button(hrtfFrame, text='HRTF Frequency Domain Visualization', state='disabled', command=lambda:freqDomainVisualHRTF())
freqDomainVisualHRTFButton.grid(row=5, column=0)

sourceFrame = tk.Frame(hrtfSourceSelectionFrame, borderwidth=10, relief='flat')
sourceFrame.grid(row=0, column=2)
selectSourceFileButton = tk.Button(sourceFrame, text='Select source file', command=lambda:selectSourceFile())
selectSourceFileLabel = tk.Label(sourceFrame, text='Source file:\n', wraplength=120)
selectSourceFileButton.grid(row=0, column=2)
selectSourceFileLabel.grid(row=1, column=2)
getSourceFileDataButton = tk.Button(sourceFrame, text='Get Source File Data', state='disabled', command=lambda:getSourceFileData())
getSourceFileDataButton.grid(row=2, column=2)
stereoToMonoButton = tk.Button(sourceFrame, text='Source File Stereo -> Mono', state='disabled', command=lambda:stereoToMono())
stereoToMonoButton.grid(row=3, column=2)


hrtfOperationsFrame = tk.Frame(topSectionFrame, borderwidth=10, relief='flat')
hrtfOperationsFrame.grid(row=1, column=1)

resampleButton = tk.Button(hrtfOperationsFrame, text='Resample', state='disabled', command=lambda:fs_resample(sig_mono, fs_s, HRIR, fs_H))
resampleButton.grid(row=0, column=1)
timeDomainConvolveButton = tk.Button(hrtfOperationsFrame, text='Time Domain Convolve', state='disabled', command=lambda:timeDomainConvolve())
timeDomainConvolveButton.grid(row=1, column=1)
exportConvolvedButton = tk.Button(hrtfOperationsFrame, text='Export Convolved', state='disabled', command=lambda:exportConvolved())
exportConvolvedButton.grid(row=2, column=1)

sectionalLabel = tk.Label(rootFrame, text='\n')
sectionalLabel.grid(row=2, column=0, columnspan=3)

sofaLabel = tk.Label(rootFrame, text="Garrett's Great\nSOFA Functions\n", font=("TkDefaultFont", str(parse_font_dict['size'] + 2), "bold"))
sofaLabel.grid(row=3, column=0, columnspan=3)

bottomSectionFrame = tk.Frame(rootFrame, borderwidth=10, relief='ridge')
bottomSectionFrame.grid(row=4, column=0, columnspan=3)
selectSOFAFileButton = tk.Button(bottomSectionFrame, text='Select SOFA File', command=lambda:selectSOFAFile())
selectSOFAFileLabel = tk.Label(bottomSectionFrame, text='SOFA file:\n', wraplength=240)
selectSOFAFileButton.grid(row=0, column=0, columnspan=3)
selectSOFAFileLabel.grid(row=1, column=0, columnspan=3)

bottomLeftFrame = tk.Frame(bottomSectionFrame, borderwidth=10, relief='flat')
bottomLeftFrame.grid(row=2, column=0)
getSOFAFileMetadataButton = tk.Button(bottomLeftFrame, text='Get SOFA File Metadata', state='disabled', command=lambda:getSOFAFileMetadata())
getSOFAFileMetadataButton.grid(row=0, column=0)
sofaMeasurementTextBox = tk.Text(bottomLeftFrame, state='disabled', height=1, width=5)
sofaMeasurementTextBox.grid(row=1, column=0)
sofaMeasurementLabel = tk.Label(bottomLeftFrame, text='Measurement Index\n(default: 0)\n')
sofaMeasurementLabel.grid(row=2, column=0)
frequencyXLimTextBox = tk.Text(bottomLeftFrame, state='disabled', height=1, width=15)
frequencyXLimTextBox.grid(row=3, column=0)
frequencyXLimLabel = tk.Label(bottomLeftFrame, text='Frequency Range (Hz)\n[start, end]')
frequencyXLimLabel.grid(row=4, column=0)
azimuthTextBox = tk.Text(bottomLeftFrame, state='disabled', height=1, width=5)
azimuthTextBox.grid(row=5, column=0)
azimuthLabel = tk.Label(bottomLeftFrame, text='Desired azimuth (in deg)')
azimuthLabel.grid(row=6, column=0)

bottomRightFrame = tk.Frame(bottomSectionFrame, borderwidth=10, relief='flat')
bottomRightFrame.grid(row=2, column=2)
getSOFAFileDimensionsButton = tk.Button(bottomRightFrame, text='Get SOFA File Dimensions', state='disabled', command=lambda:getSOFAFileDimensions())
getSOFAFileDimensionsButton.grid(row=0, column=0)
sofaEmitterTextBox = tk.Text(bottomRightFrame, state='disabled', height=1, width=5)
sofaEmitterTextBox.grid(row=1, column=0)
sofaEmitterLabel = tk.Label(bottomRightFrame, text='Emitter\n(default: 1)\n')
sofaEmitterLabel.grid(row=2, column=0)
magnitudeYLimTextBox = tk.Text(bottomRightFrame, state='disabled', height=1, width=15)
magnitudeYLimTextBox.grid(row=3, column=0)
magnitudeYLimLabel = tk.Label(bottomRightFrame, text='Magnitude (dB)\n[start, end]')
magnitudeYLimLabel.grid(row=4, column=0)
elevationTextBox = tk.Text(bottomRightFrame, state='disabled', height=1, width=5)
elevationTextBox.grid(row=5, column=0)
elevationLabel = tk.Label(bottomRightFrame, text='Desired elevation (in deg)')
elevationLabel.grid(row=6, column=0)

sofaViewButton = tk.Button(bottomSectionFrame, text='View SOFA File', state='disabled', command=lambda:displaySOFAFile(frequencyXLimTextBox.get(1.0, 'end-1c'), magnitudeYLimTextBox.get(1.0, 'end-1c'), sofaMeasurementTextBox.get(1.0, 'end-1c'), sofaEmitterTextBox.get(1.0, 'end-1c')))
sofaViewButton.grid(row=3, column=0, columnspan=3)
sofaDisplayButton = tk.Button(bottomSectionFrame, text='Render Source with SOFA file', state='disabled', command=lambda:manualSOFADisplay(azimuthTextBox.get(1.0, 'end-1c'), elevationTextBox.get(1.0, 'end-1c'), source_file))
sofaDisplayButton.grid(row=4, column=0, columnspan=3)

tutorialButton = tk.Button(rootFrame, text='Help', command=lambda:createHelpWindow())
tutorialButton.grid(row=5, column=0, sticky='W')

quitButton = tk.Button(rootFrame, text='Quit', command=lambda:root.destroy())
quitButton.grid(row=5, column=2, sticky='E')

class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def show_tooltip(self, text): #https://stackoverflow.com/questions/20399243/display-message-when-hovering-over-something-with-mouse-cursor-in-python
        self.text = text
        if self.tipwindow  or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        tooltip_label = tk.Label(tw, text=self.text, justify='left', background='#ffffe0', relief='solid', borderwidth=0, font="TkDefaultFont 8 normal")
        tooltip_label.pack(ipadx=1)

    def hide_tooltip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def create_tooltip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.show_tooltip(text)
    def leave(event):
        toolTip.hide_tooltip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

def shorten_file_name(old_file_name):
    if len(old_file_name) > 17:
        old_file_name = old_file_name
        new_file_name = str(old_file_name[:15]) + "..."
    else:
        new_file_name = old_file_name
    return new_file_name

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
        selectHRTFFileLabel.config(text='HRTF file:\n' + shorten_file_name(hrtf_file_print[len(hrtf_file_print)-1]))
        create_tooltip(selectHRTFFileLabel, text=str(hrtf_file))
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
        selectSourceFileLabel.config(text='Source file:\n' + shorten_file_name(source_file_print[len(source_file_print)-1]))
        create_tooltip(selectSourceFileLabel, text=str(source_file))
        getSourceFileDataButton.config(state='active')
        stereoToMonoButton.config(state='active')
        resampleButton.config(state='disabled')
        [sig,fs_s] = sf.read(source_file)
    else:
        return

def getHRTFFileData():
    newWindow = tk.Toplevel(root)
    centered_window(newWindow)
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
    centered_window(newWindow)
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
    # print(HRIR)
    # print(HRTF)
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
        centered_window(newWindow)
        newWindow.geometry('350x100')
        newWindow.title('Stereo -> Mono')
        windowTitleStereoToMonoLabel = tk.Label(newWindow, text='\n' + 'New source data dimensions: ' + str(sig_mono.shape) + '\n')
        windowTitleStereoToMonoLabel.grid(row=0, column=1)
    else:
        sig_mono = sig
        newWindow = tk.Toplevel(root)
        centered_window(newWindow)
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
    centered_window(newWindow)
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

    newWindow = tk.Toplevel(root)
    centered_window(newWindow)
    newWindow.minsize(100, 100)
    if os.path.exists(str(source_file_export_name[:-4]) + '-' + str(hrtf_file_export_name[:-4]) + '-export.wav'):
        newWindow.title('Export Successful')
        windowTitleExportSuccessLabel = tk.Label(newWindow, text="\nFile successfully exported as:\n"+str(source_file_export_name[:-4]) + '-' + str(hrtf_file_export_name[:-4]) + '-export.wav')
        windowTitleExportSuccessLabel.grid(row=0, column=1)

    else:
        newWindow.title('Export Failed')
        windowTitleExportFailLabel = tk.Label(newWindow, text="File export failed.")
        windowTitleExportFailLabel.grid(row=0, column=1)


def selectSOFAFile():
    global sofa_file
    global sofa_file_print
    sofa_file = filedialog.askopenfilename()
    if sofa_file:
        sofa_file_print = sofa_file.split('/')
        selectSOFAFileLabel.config(text='SOFA file:\n' + shorten_file_name(sofa_file_print[len(sofa_file_print)-1]))
        create_tooltip(selectSOFAFileLabel, text=str(sofa_file))
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
    centered_window(newWindow)
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
    centered_window(newWindow)
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
        centered_window(newWindow)
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
    centered_window(newWindow)
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

def clearWidgets(frame_to_clear):
    for widget in frame_to_clear.winfo_children():
        widget.destroy()

def createHelpWindow():
    global tutorialWindow
    global tutorialWindowContentFrame
    tutorialWindow = tk.Toplevel(root)
    tutorialWindow.grid_columnconfigure(0, weight=1)
    tutorialWindow.grid_rowconfigure(0, weight=1)
    tutorialWindow.minsize(650, 400)
    tutorialWindow.title('Help')
    tutorialWindowContentFrame = tk.Frame(tutorialWindow, borderwidth=10, relief='flat')
    tutorialWindowContentFrame.grid(row=0, column=0)
    hrtfHelpPage()

def hrtfHelpPage():
    clearWidgets(tutorialWindowContentFrame)

    hrtfTitleTutorialLabel = tk.Label(tutorialWindowContentFrame, text='HRTF Functions\n', font=("TkDefaultFont", str(parse_font_dict['size'] + 4), "bold"))
    hrtfTitleTutorialLabel.grid(row=0, column=1)

    selectHRTFTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Select HRTF File (.wav)"\nPresents dialogue box for selecting HRTF.\nOnly takes .wav files. Expects an IR.\n')
    selectHRTFTutorialLabel.grid(row=1, column=0)
    getHRTFFileDataTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Get HRTF File Data"\nPresents info about HRTF file, including:\n- Sample rate\n- Data dimension (num samples, num channels)\n- Ability to play loaded HRTF file.\n')
    getHRTFFileDataTutorialLabel.grid(row=2, column=0)
    hrtfTimeDomainVisualizationTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"HRTF Time Domain Visualization"\nTime domain plot of loaded HRTF.\n')
    hrtfTimeDomainVisualizationTutorialLabel.grid(row=3, column=0)
    hrtfFrequencyDomainVisualizationTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"HRTF Frequency Domain Visualization"\nFrequency domain plot of loaded HRTF.\n')
    hrtfFrequencyDomainVisualizationTutorialLabel.grid(row=4, column=0)

    selectSourceFileTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Select Source File"\nPresents dialogue box for selecting a source file.\nOnly takes .wav files.\nUsed for convolving with loaded HRTF.\n')
    selectSourceFileTutorialLabel.grid(row=1, column=2)
    getSourceFileDataTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Get Source File Data"\nPresents info about source file, including:\n- Sample rate\n- Data dimension (num samples, num channels)\n- Ability to play loaded selected file.\n')
    getSourceFileDataTutorialLabel.grid(row=2, column=2)
    stereoToMonoTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Source File Stereo -> Mono"\nDownmixes stereo file into mono\nfor convenience with convolving.\n')
    stereoToMonoTutorialLabel.grid(row=3, column=2)

    resampleTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Resample"\nResamples source file to match sample rate of loaded HRTF file.\nResampled source file is held in memory, not exported.\n')
    resampleTutorialLabel.grid(row=5, column=1)
    timeDomainConvolveTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Time Domain Convolve"\nTime domain convolves loaded source file with loaded HRTF file.\nSource file should either match HRTF file sample rate,\nor have been resampled with the button above.\nConvolved file is held in memory, not exported.\n')
    timeDomainConvolveTutorialLabel.grid(row=6, column=1)
    exportConvolvedTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Export Convolved"\nExports the time domain convolved file loaded in memory.\nFile naming convention is:\n[Source File Name]-[HRTF File Name]-export.wav\n')
    exportConvolvedTutorialLabel.grid(row=7, column=1)

    nextButton = tk.Button(tutorialWindowContentFrame, text='Next ->', command=lambda:sofaHelpPage())
    nextButton.grid(row=8, column=2, sticky='E')

def sofaHelpPage():
    clearWidgets(tutorialWindowContentFrame)
    sofaTitleTutorialLabel = tk.Label(tutorialWindowContentFrame, text='SOFA Functions\n', font=("TkDefaultFont", str(parse_font_dict['size'] + 4), "bold"))
    sofaTitleTutorialLabel.grid(row=0, column=1)

    selectSOFAFileTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Select SOFA File"\nPresents dialogue box for selecting .SOFA file.\n')
    selectSOFAFileTutorialLabel.grid(row=1, column=1)
    getSOFAFileMetadataTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Get SOFA File Metadata"\nPresents metadata embedded in the loaded SOFA file.\nFollows SOFA convention.\n')
    getSOFAFileMetadataTutorialLabel.grid(row=2, column=0)
    sofaMeasurementTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Measurement index"\nChoose the measurement index to be used when viewing plot data.\nCheck .SOFA file dimensions for measurement indices.\n')
    sofaMeasurementTutorialLabel.grid(row=3, column=0)
    frequencyXLimTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Frequency Range (Hz)"\nConfigurable range for x-axis of .SOFA file plot.\n')
    frequencyXLimTutorialLabel.grid(row=4, column=0)
    desiredAzimuthTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Desired azimuth (in deg)"\nSelectable azimuth for viewing .SOFA file plot.\n')
    desiredAzimuthTutorialLabel.grid(row=5, column=0)

    getSOFAFileDimensionsTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Get SOFA File Dimensions"\nPresents info about the SOFA convention dimensions\nwithin the .SOFA file.\n')
    getSOFAFileDimensionsTutorialLabel.grid(row=2, column=2)
    sofaEmitterTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Emitter"\nChoose the emitter to be used when viewing plot data.\nCheck .SOFA file dimensions for emitters.\n')
    sofaEmitterTutorialLabel.grid(row=3, column=2)
    magnitudeYLimTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Magnitude (dB)"\nConfigurable range for y-axis of .SOFA file plot.\n')
    magnitudeYLimTutorialLabel.grid(row=4, column=2)
    desiredElevationTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Desired elevation (in deg)"\nSelectable elevation for viewing .SOFA file plot.\n')
    desiredElevationTutorialLabel.grid(row=5, column=2)

    viewSOFAFileTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"View SOFA File"\nTakes the above selected values\nand presents a 3D view of the .SOFA file,\n in addition to individual measurements\nfrom the .SOFA file.\n')
    viewSOFAFileTutorialLabel.grid(row=6, column=1)
    renderSOFATutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Render Source with SOFA File"\nConvolves the source file with\nthe desired values in the .SOFA file.\n')
    renderSOFATutorialLabel.grid(row=7, column=1)

    prevButton = tk.Button(tutorialWindowContentFrame, text='<- Previous', command=lambda:hrtfHelpPage())
    prevButton.grid(row=8, column=0, sticky='W')

root.mainloop()