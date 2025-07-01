# much of this is sourced from andrea genovese
# please find him here: https://andreagenovese.com/
# and his wonderful tutorial about hrtf processing here: https://www.youtube.com/watch?v=a4mpK_2koR4

# also please keep in mind that this isn't supposed to be "efficient" or "clean" or "lightweight".
# this is meant to be the most brute force way to do all my hrtf processing in one file with one interface. 

import os
import tempfile
import sys

os.environ['NUMBA_CACHE_DIR'] = str(tempfile.gettempdir())
os.environ['LIBROSA_CACHE_DIR'] = str(tempfile.gettempdir())
print(sys.path)

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf # <- read audio
import sofa # <- read SOFA HRTFs
import librosa # <- resample function
import pygame # <- for playing audio files directly.
import webbrowser
import tkinter as tk
from tkinter import filedialog
from tkinter import font
from scipy import signal # <- fast convolution function
from scipy.io import wavfile # <- used for spectrogram
# from PySide6 import QtCore, QtWidgets, QtGui
# from PyQt6 import QtWidgets # <- for displaying matplotlib in standalone
# import pyinstaller # <- only used for generating requirements.txt for my own .venv on multiple machines
# try:
#     import pyi_splash # <- only for pyinstaller.
#     pyi_splash.close()
# except:
#     pass

if "NUITKA_ONEFILE_PARENT" in os.environ:
    
    splash_filename = os.path.join(
        tempfile.gettempdir(),
        "onefile_%d_splash_feedback.tmp" % int(os.environ["NUITKA_ONEFILE_PARENT"]),
    )

    if os.path.exists(splash_filename):
        os.unlink(splash_filename)

librosa.cache.clear()

source_file = None

class ToolTip(object): #https://stackoverflow.com/questions/20399243/display-message-when-hovering-over-something-with-mouse-cursor-in-python
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def show_tooltip(self, text):
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

def centered_window(window): #https://www.geeksforgeeks.org/how-to-center-a-window-on-the-screen-in-tkinter/
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

def create_tooltip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.show_tooltip(text)
    def leave(event):
        toolTip.hide_tooltip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

def shorten_file_name(old_filename, num_shown_char):
    # TODO: REVISIT, you know better ways to do this.
    """
    Takes a long string and shortens it to the length provided before appending ellipses.

    Args:
        old_filename (str): Full length filename to be shortened.
        num_shown_char (int): Number of letters from the old filename to show before inserting ellipses.

    Returns:
        str: Shortened version of the full filename to be shown with ellipses added.
    """    
    if len(old_filename) > 17:
        old_filename = old_filename
        new_filename = str(old_filename[:int(num_shown_char)]) + "..."
    else:
        new_filename = old_filename
    return new_filename

def playAudio(path_to_audio_file):
    """
    Uses pygame to play audio in the app.

    Args:
        path_to_audio_file (str): Path to audio file to be loaded.
    """    
    pygame.mixer.init()
    pygame.mixer.music.load(path_to_audio_file)
    pygame.mixer.music.play()

def selectHRTFFile():
    global hrtf_file
    global hrtf_file_print
    global HRIR
    global fs_H
    hrtf_file = filedialog.askopenfilename(filetypes=[('.wav files', '.wav')])
    try:
        if hrtf_file:
            [HRIR,fs_H] = sf.read(hrtf_file)
    except RuntimeError:
        errorWindow('\nError loading file:\n\n' + str(os.path.basename(hrtf_file)) + '\n\nRuntime error.\nDoes this file exist on the local drive?', title='Error', width=300, height=200, tooltip_text=hrtf_file)
        return
    else:
        if hrtf_file:
            hrtf_file_print = hrtf_file.split('/')
            selectHRTFFileLabel.config(text='HRTF file:\n' + shorten_file_name(hrtf_file_print[len(hrtf_file_print)-1], 15))
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
    source_file = filedialog.askopenfilename(filetypes=[('.wav files', '.wav')])
    # check to see if the selected file is on the drive (in the case of network drives, for example)
    try:
        if source_file:
            [sig,fs_s] = sf.read(source_file)
    except RuntimeError:
        errorWindow('\nError loading file:\n\n' + str(os.path.basename(source_file)) + '\n\nRuntime error.\nDoes this file exist on the local drive?', title='Error', width=300, height=200, tooltip_text=source_file)
        return
    else:
        if source_file == '':
            return
        source_file_print = source_file.split('/')
        selectSourceFileLabel.config(text='Source file:\n' + shorten_file_name(source_file_print[len(source_file_print)-1], 15))
        create_tooltip(selectSourceFileLabel, text=str(source_file))
        getSourceFileDataButton.config(state='active')
        stereoToMonoButton.config(state='active')
        spectrogramButton.config(state='active')
        resampleButton.config(state='disabled')
        [sig,fs_s] = sf.read(source_file)

def getHRTFFileData():
    newWindow = tk.Toplevel(root)
    newWindow.iconphoto(False, icon_photo)
    centered_window(newWindow)
    newWindow.minsize(300, 120)
    newWindow.title('HRTF File Data')
    windowTitleHRTFData = tk.Label(newWindow, text='\n' + hrtf_file_print[len(hrtf_file_print)-1] + '\n')
    windowTitleHRTFData.pack()
    create_tooltip(windowTitleHRTFData, hrtf_file)
    hrtfSampleRateLabel = tk.Label(newWindow, text='Sample rate: ' + str(fs_H))
    hrtfSampleRateLabel.pack()
    hrtfDataDimLabel = tk.Label(newWindow, text='Data dimensions: ' + str(HRIR.shape))
    hrtfDataDimLabel.pack()
    hrtfPlayFileButton = tk.Button(newWindow, text='Play HRTF', command=lambda:playAudio(hrtf_file))
    hrtfPlayFileButton.pack()
    hrtfPauseFileButton = tk.Button(newWindow, text='Pause HRTF', command=lambda:pygame.mixer.music.pause())
    hrtfPauseFileButton.pack()

def getSourceFileData():
    newWindow = tk.Toplevel(root)
    newWindow.iconphoto(False, icon_photo)
    centered_window(newWindow)
    newWindow.minsize(300, 120)
    newWindow.title('Source File Data')
    windowTitleSourceData = tk.Label(newWindow, text='\n' + source_file_print[len(source_file_print)-1] + '\n')
    windowTitleSourceData.pack()
    create_tooltip(windowTitleSourceData, source_file)
    sourceSampleRateLabel = tk.Label(newWindow, text='Sample rate: ' + str(fs_s))
    sourceSampleRateLabel.pack()
    sourceDataDimLabel = tk.Label(newWindow, text='Data dimensions: ' + str(sig.shape))
    sourceDataDimLabel.pack()
    sourcePlayFileButton = tk.Button(newWindow, text='Play Source File', command=lambda:playAudio(source_file))
    sourcePlayFileButton.pack()
    sourcePauseFileButton = tk.Button(newWindow, text='Pause Source File', command=lambda:pygame.mixer.music.pause())
    sourcePauseFileButton.pack()

def timeDomainVisualHRTF():
    plt.figure(num=str('Time Domain for ' + os.path.basename(hrtf_file)))
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
    HRTF_mag = (2/nfft)*np.abs(HRTF[0:int(len(HRTF)/2)+1,:])
    HRTF_mag_dB = 20*np.log10(HRTF_mag)

    f_axis = np.linspace(0,fs_H/2,len(HRTF_mag_dB))
    plt.figure(num=str('Frequency Domain for ' + os.path.basename(hrtf_file)))
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
        newWindow.iconphoto(False, icon_photo)
        centered_window(newWindow)
        newWindow.geometry('350x100')
        newWindow.minsize(350,100)
        newWindow.title('Stereo -> Mono')
        windowTitleStereoToMonoLabel = tk.Label(newWindow, text='\n' + 'New source data dimensions: ' + str(sig_mono.shape) + '\n')
        windowTitleStereoToMonoLabel.pack()
    else:
        sig_mono = sig
        newWindow = tk.Toplevel(root)
        newWindow.iconphoto(False, icon_photo)
        centered_window(newWindow)
        newWindow.geometry('350x100')
        newWindow.minsize(350,100)
        newWindow.title('Stereo -> Mono')
        windowTitleStereoToMonoLabel = tk.Label(newWindow, text='\n' + 'Source file already mono.' + '\n')
        windowTitleStereoToMonoLabel.pack()
    resampleButton.config(state='active')
    
def fs_resample(s1, f1, s2, f2):
    """
    For two signals that have differing sample rates, resample the lower to meet the higher.

    Args:
        s1 (ndarray): First signal.
        f1 (int): Sampling rate of s1.
        s2 (ndarray): Second signal.
        f2 (int): Sampling rate of s2.

    Returns:
        int, int, int, int: Resampled signal 1, signal 1's resampled sampling rate, resampled signal 2, signal 2's resampled sampling rate.
    """    
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
    resampleWindow.iconphoto(False, icon_photo)
    centered_window(resampleWindow)
    resampleWindow.geometry('250x150')
    resampleWindow.minsize(250, 150)
    resampleWindow.title('Resample')
    windowTitleResampleLabel = tk.Label(resampleWindow, text='\n' + 'Resampled at: ' + str(fmax) + 'Hz' + '\n')
    windowTitleResampleLabel.pack()
    resampleSignalSourceDimensionsLabel = tk.Label(resampleWindow, text='Signal/source dimensions: ' + str(sig_mono.shape))
    resampleSignalSourceDimensionsLabel.pack()
    resampleHRIRDimensionsLabel = tk.Label(resampleWindow, text='HRIR Dimensions: ' + str(HRIR.shape))
    resampleHRIRDimensionsLabel.pack()
    timeDomainConvolveButton.config(state='active')
    return s1, f1, s2, f2

def timeDomainConvolve():
    global Bin_Mix
    s_L = np.convolve(sig_mono,HRIR[:,0])
    s_R = np.convolve(sig_mono,HRIR[:,1])

    Bin_Mix = np.vstack([s_L,s_R]).transpose()
    newWindow = tk.Toplevel(root)
    newWindow.iconphoto(False, icon_photo)
    centered_window(newWindow)
    newWindow.geometry('250x100')
    newWindow.minsize(250, 100)
    newWindow.title('Time Domain Convolve')
    timeDomainConvolveWindowLabel = tk.Label(newWindow, text='\n' + 'Data dimensions: ' + str(Bin_Mix.shape) + '\n')
    timeDomainConvolveWindowLabel.pack()
    exportConvolvedButton.config(state='active')

# freq domain convolution goes here at some point

def exportConvolved():
    source_file_location = os.path.join(*source_file_print[:-1])
    export_directory = filedialog.askdirectory(title='Select Save Directory', initialdir=source_file_location)
    source_file_export_name = source_file_print[(len(source_file_print)-1)]
    hrtf_file_export_name = hrtf_file_print[(len(hrtf_file_print)-1)]

    if export_directory:
        convolved_filename = str(source_file_export_name[:-4]) + '-' + str(hrtf_file_export_name[:-4]) + '-export.wav'
        export_filename = os.path.join(str(export_directory), str(convolved_filename))
        sf.write(export_filename, Bin_Mix, fs_s)
    if not export_directory:
        return -1

    newWindow = tk.Toplevel(root)
    newWindow.iconphoto(False, icon_photo)
    centered_window(newWindow)
    newWindow.geometry('400x100')
    newWindow.minsize(400, 100)
    if os.path.exists(export_filename):
        newWindow.title('Export Successful')
        windowTitleExportSuccessLabel = tk.Label(newWindow, text="\nFile successfully exported as:\n"+str(source_file_export_name[:-4]) + '-' + str(hrtf_file_export_name[:-4]) + '-export.wav')
        windowTitleExportSuccessLabel.pack()
        create_tooltip(windowTitleExportSuccessLabel, str(export_filename))
    else:
        newWindow.title('Export Failed')
        windowTitleExportFailLabel = tk.Label(newWindow, text="File export failed.")
        windowTitleExportFailLabel.pack()


def selectSOFAFile():
    global sofa_file
    global sofa_file_print
    sofa_file = filedialog.askopenfilename(filetypes=[('.SOFA files', '.sofa')])
    try:
        if sofa_file:
            metadata_test = sofa.Database.open(sofa_file).Metadata.list_attributes()
    except OSError:
        errorWindow('\nError loading file:\n\n' + str(os.path.basename(sofa_file)) + '\n\nOS error.\nDoes this file exist on the local drive?\n\nAlternatively, does this SOFA file\ncontain correct metadata?', title='Error', width=300, height=250, tooltip_text=sofa_file)
        return
    else:
        if sofa_file:
            sofa_file_print = sofa_file.split('/')
            selectSOFAFileLabel.config(text='SOFA file:\n' + shorten_file_name(sofa_file_print[len(sofa_file_print)-1], 20))
            create_tooltip(selectSOFAFileLabel, text=str(sofa_file))
            getSOFAFileMetadataButton.config(state='active')
            getSOFAFileDimensionsButton.config(state='active')
            sofaDisplayButton.config(state='active')
            sofaViewButton.config(state='active')
            sofaSaveButton.config(state='active')
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
    newWindow.iconphoto(False, icon_photo)
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

    v.config(command=windowSOFAMetadataLabel.yview)

def getSOFAFileDimensions():
    newWindow = tk.Toplevel(root)
    newWindow.iconphoto(False, icon_photo)
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

def exportSOFAConvolved(source_file_name, angle_label, elev_label, audioContent, sofa_positions, samplerate=48000):
    sofa_file_location = os.path.join(*sofa_file_print[:-1])
    export_directory = filedialog.askdirectory(title='Select Save Directory', initialdir=sofa_file_location)

    newWindow = tk.Toplevel(root)
    newWindow.iconphoto(False, icon_photo)
    newWindow.minsize(500, 150)
    newWindow.title('SOFA Rendering')
    windowHRTFInfoLabel = tk.Label(newWindow, text='\nUsing HRTF set: ' + sofa_file_print[len(sofa_file_print)-1])
    windowHRTFInfoLabel.pack()
    create_tooltip(windowHRTFInfoLabel, str(sofa_file))
    centered_window(newWindow)
    try:
        windowSourceDistanceLabel = tk.Label(newWindow, text='Source distance is: ' + str(sofa_positions[0,2]) + 'meters\n')
        windowSourceDistanceLabel.pack()
    except:
        windowSourceNoDistanceLabel = tk.Label(newWindow, text='No distance information available\n')
        windowSourceNoDistanceLabel.pack()

    if export_directory:
        export_filename = os.path.join(str(export_directory), str((sofa_file_print[len(sofa_file_print)-1])[:-5]))
        sf.write(export_filename + '-' + str(source_file_name[:-4]) + '-azi_' + str(angle_label) + '-elev_' + str(elev_label) + '-export.wav', audioContent, samplerate=samplerate)
        windowSOFAInfoLabel = tk.Label(newWindow, text='Source: ' + str(source_file_print[len(source_file_print)-1]) + '\nrendered at azimuth ' + str(angle_label) + ' and elevation ' + str(elev_label))
        windowSOFAInfoLabel.pack()
        create_tooltip(windowSOFAInfoLabel, str(export_filename + '-' + str(source_file_name[:-4]) + '-azi_' + str(angle_label) + '-elev_' + str(elev_label) + '-export.wav'))
    if not export_directory:
        windowSOFAInfoLabel = tk.Label(newWindow, text='Not rendered: Export directory not given.')
        windowSOFAInfoLabel.pack()

def plot_coordinates(coords, plot_title, in_sofa_file):
    x0 = coords
    n0 = coords
    fig = plt.figure(figsize=(10,7))
    window_title = 'SOFA Source Positions for ' + os.path.basename(in_sofa_file)
    fig.canvas.manager.set_window_title(str(window_title))
    ax = fig.add_subplot(111, projection='3d')
    q = ax.quiver(x0[:, 0], x0[:, 1], x0[:, 2], 
                  n0[:, 0], n0[:, 1], n0[:, 2], 
                  length=0.1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(plot_title)
    return q

def displaySOFAFile(xlim, ylim, measurement=0, emitter=1):
    if not xlim:
        xlim = '20, 20000'
    if not ylim:
        ylim = '-150, 0'
    if measurement == '':
        measurement = 0
    if emitter == '':
        emitter = 1
    SOFA_HRTF = sofa.Database.open(sofa_file)

    # plot source coordinates
    source_positions = SOFA_HRTF.Source.Position.get_values(system='cartesian')
    sofa_positions_plot_title = 'Source positions for: ' + str(os.path.basename(sofa_file))
    plot_coordinates(source_positions, str(sofa_positions_plot_title), sofa_file)

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

    hrir_plot_title = 'Head-Related Impulse Response for: ' + os.path.basename(sofa_file)
    plt.figure(figsize=(15, 5), num=str(hrir_plot_title))
    for receiver in np.arange(SOFA_HRTF.Dimensions.R):
        plt.plot(t, SOFA_HRTF.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))
        legend.append('Receiver {0}'.format(receiver))
    plt.title('{0}: HRIR at M={1} for emitter {2}'.format(os.path.basename(sofa_file), measurement, emitter))
    plt.legend(legend)
    plt.xlabel('$t$ in s')
    plt.ylabel(r'$h(t)$')
    plt.grid()

    # not so sure this is the best way to do this but it's probably fine
    nfft = len(SOFA_HRTF.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))*8
    HRTF = np.fft.fft(SOFA_HRTF.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}),n=nfft, axis=0)
    HRTF_mag = (2/nfft)*np.abs(HRTF[0:int(len(HRTF)/2)+1])
    HRTF_mag_dB = 20*np.log10(HRTF_mag)

    hrtf_plot_title = 'Head-Related Transfer Function for: ' + os.path.basename(sofa_file)
    plt.figure(figsize=(15, 5), num=str(hrtf_plot_title))
    f_axis = np.linspace(0,(SOFA_HRTF.Data.SamplingRate.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))/2,len(HRTF_mag_dB))
    plt.semilogx(f_axis, HRTF_mag_dB)
    ax = plt.gca()
    ax.set_xlim([int(xlim_start), int(xlim_end)])
    ax.set_ylim([int(ylim_start), int(ylim_end)])
    plt.grid()
    plt.grid(which='minor', color="0.9")
    plt.title('{0}: HRTF at M={1} for emitter {2}'.format(os.path.basename(sofa_file), measurement, emitter))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend(['Left','Right'])
    plt.show()

    SOFA_HRTF.close()
    plt.close()
    
def saveSOFAFile(xlim, ylim, measurement=0, emitter=1):
    plt.close()
    export_directory = filedialog.askdirectory(title='Select Save Directory', initialdir=os.path.dirname(sofa_file))
    if not export_directory:
        errorWindow(error_message='Directory not given.')
        return -1
    export_directory = os.path.join(export_directory, os.path.basename(sofa_file))
    export_directory = export_directory + '-measurements'
    try:
        os.mkdir(export_directory)
    except FileExistsError:
        pass
    
    if not xlim:
        xlim = '20, 20000'
    if not ylim:
        ylim = '-150, 0'
    if measurement == '':
        measurement = 0
    if emitter == '':
        emitter = 1
    SOFA_HRTF = sofa.Database.open(sofa_file)

    # plot source coordinates
    x0 = SOFA_HRTF.Source.Position.get_values(system='cartesian')
    n0 = SOFA_HRTF.Source.Position.get_values(system='cartesian')
    fig = plt.figure(figsize=(10,7))
    window_title = 'SOFA Source Positions for ' + os.path.basename(sofa_file)
    fig.canvas.manager.set_window_title(str(window_title))
    ax = fig.add_subplot(111, projection='3d')
    q = ax.quiver(x0[:, 0], x0[:, 1], x0[:, 2], 
                  n0[:, 0], n0[:, 1], n0[:, 2], 
                  length=0.1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Source positions for: ' + str(os.path.basename(sofa_file)))
    unedited_sofa_filename = os.path.basename(sofa_file)
    for letter in unedited_sofa_filename:
        if letter == '':
            letter = '_'
    sofa_pos_indiv_filename = 'SOFA_Source_Positions_for_' + unedited_sofa_filename + '.png'
    sofa_pos_export_filename = os.path.join(str(export_directory), sofa_pos_indiv_filename)
    plt.savefig(sofa_pos_export_filename)

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

    hrir_window_title = 'Head-Related Impulse Response for ' + os.path.basename(sofa_file)
    plt.figure(figsize=(15, 5), num=str(hrir_window_title))
    for receiver in np.arange(SOFA_HRTF.Dimensions.R):
        plt.plot(t, SOFA_HRTF.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))
        legend.append('Receiver {0}'.format(receiver))
    plt.title('{0}: HRIR at M={1} for emitter {2}'.format(os.path.basename(sofa_file), measurement, emitter))
    plt.legend(legend)
    plt.xlabel('$t$ in s')
    plt.ylabel(r'$h(t)$')
    plt.grid()
    sofa_hrir_indiv_filename = 'Head-Related_Impulse_Response_for_' + unedited_sofa_filename + '.png'
    sofa_hrir_export_filename = os.path.join(str(export_directory), sofa_hrir_indiv_filename)
    plt.savefig(sofa_hrir_export_filename)
    

    # not so sure this is the best way to do this but it's probably fine
    nfft = len(SOFA_HRTF.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))*8
    HRTF = np.fft.fft(SOFA_HRTF.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}),n=nfft, axis=0)
    HRTF_mag = (2/nfft)*np.abs(HRTF[0:int(len(HRTF)/2)+1])
    HRTF_mag_dB = 20*np.log10(HRTF_mag)

    hrtf_window_title = 'Head-Related Transfer Function for ' + os.path.basename(sofa_file)
    plt.figure(figsize=(15, 5), num=str(hrtf_window_title))
    f_axis = np.linspace(0,(SOFA_HRTF.Data.SamplingRate.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))/2,len(HRTF_mag_dB))
    plt.semilogx(f_axis, HRTF_mag_dB)
    ax = plt.gca()
    ax.set_xlim([int(xlim_start), int(xlim_end)])
    ax.set_ylim([int(ylim_start), int(ylim_end)])
    plt.grid()
    plt.grid(which='minor', color="0.9")
    plt.title('{0}: HRTF at M={1} for emitter {2}'.format(os.path.basename(sofa_file), measurement, emitter))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend(['Left','Right'])
    # plt.show()
    sofa_hrtf_indiv_filename = 'Head-Related_Transfer_Function_for_' + unedited_sofa_filename + '.png'
    sofa_hrtf_export_filename = os.path.join(str(export_directory), sofa_hrtf_indiv_filename)
    plt.savefig(sofa_hrtf_export_filename)

    SOFA_HRTF.close()
    plt.close()
    
    messageWindow('Successfully exported to:\n' + (os.path.basename(sofa_file) + '-measurements'), title='Export Successful', width=400, tooltip_text=export_directory)


def manualSOFADisplay(angle, elev, source_file, target_fs=48000):
    global Stereo3D
    try:
        isinstance(source_file, str)
    except NameError:
        errorWindow('NameError: Missing source file.')
        return
    except TypeError:
        errorWindow('TypeError: Missing source file.')
        return
    except not source_file:
        errorWindow('Empty Source File: Missing source file.')
        return
    except type(source_file) == None:
        errorWindow('NoneType: Missing source file.')
        return
    else:
        if not source_file:
            errorWindow('Missing source file.')
        
        if not angle:
            angle = 0
        
        if not elev:
            elev = 0

        # init
        SOFA_HRTF = sofa.Database.open(sofa_file)
        sofa_fs_H = SOFA_HRTF.Data.SamplingRate.get_values()[0]
        sofa_positions = SOFA_HRTF.Source.Position.get_values(system='spherical')
        SOFA_H = np.zeros([SOFA_HRTF.Dimensions.N,2])
        Stereo3D = np.zeros([SOFA_HRTF.Dimensions.N,2])

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
            # print("---\n** Recalc SR **\nSOFA file SR =", sofa_fs_H, "\nTarget SR =", target_fs)
            # print("---\nOld shape =", SOFA_H.shape)
            SOFA_H = librosa.core.resample(SOFA_H.transpose(), orig_sr=int(sofa_fs_H), target_sr=int(target_fs), fix=True).transpose()
            # print("New shape =", SOFA_H.shape)
        
        # pick a source (already picked)
        [source_x, fs_x] = sf.read(source_file)
        if len(source_x.shape)>1:
            if source_x.shape[1] > 1:
                source_x = np.mean(source_x, axis=1)
        if fs_x != target_fs:
            # print("** Recalc SR **\nAudio file SR =", fs_x, "\nTarget SR =", target_fs)
            source_x = librosa.core.resample(source_x.transpose(), orig_sr=int(fs_x), target_sr=int(target_fs), fix=True).transpose()

        rend_L = signal.fftconvolve(source_x,SOFA_H[:,0])
        rend_R = signal.fftconvolve(source_x,SOFA_H[:,1])
        M = np.max([np.abs(rend_L), np.abs(rend_R)])
        if len(Stereo3D) < len(rend_L):
            diff = len(rend_L) - len(Stereo3D)
            Stereo3D = np.append(Stereo3D, np.zeros([diff,2]),0)
        Stereo3D[0:len(rend_L),0] += (rend_L/M)
        Stereo3D[0:len(rend_R),1] += (rend_R/M)

        exportSOFAConvolved(str(source_file_print[len(source_file_print)-1]), angle_label, elev_label, Stereo3D, sofa_positions, int(target_fs))

        return Stereo3D

def spectrogram(audio_file, start_msSV, end_msSV, dynamic_range_minSV, dynamic_range_maxSV, plot_titleSV):
    start_ms = start_msSV.get()
    end_ms = end_msSV.get()
    plot_title = plot_titleSV.get()
    dynamic_range_min = dynamic_range_minSV.get()
    dynamic_range_max = dynamic_range_maxSV.get()

    if not plot_title:
        plot_title = os.path.basename(audio_file)
    
    sr, samples = wavfile.read(str(audio_file))

    if len(samples.shape) > 1:
        samples = samples.transpose()[0]

    if not start_ms:
        start_ms = '0'

    if not end_ms:
        end_ms = str((len(samples) / sr) * 1000)
    
    start_in_samples = (float(start_ms)/1000) * sr
    end_in_samples = (float(end_ms)/1000) * sr

    if start_in_samples >= end_in_samples:
        errorWindow(error_message='End time cannot be equal to or less than start time.', width=400)
        return
    
    start_in_samples = int(start_in_samples)
    end_in_samples = int(end_in_samples)
    rebound_samples = samples[start_in_samples:end_in_samples]

    f, t, spectrogram = signal.spectrogram(rebound_samples, sr)

    fig, ax = plt.subplots()
    if dynamic_range_min and dynamic_range_max:
        p = ax.pcolormesh(t, f, 10*np.log10(spectrogram), vmin=int(dynamic_range_min), vmax=int(dynamic_range_max), shading='auto')
    elif not dynamic_range_min and dynamic_range_max:
        p = ax.pcolormesh(t, f, 10*np.log10(spectrogram), vmax=int(dynamic_range_max), shading='auto')
    elif dynamic_range_min and not dynamic_range_max:
        p = ax.pcolormesh(t, f, 10*np.log10(spectrogram), vmin=int(dynamic_range_min), shading='auto')
    else:
        p = ax.pcolormesh(t, f, 10*np.log10(spectrogram), shading='auto')

    ax.set_ylim(1, int(sr/2))
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')

    fig.colorbar(p, label='Intensity (dB)')
    fig.canvas.manager.set_window_title(str('Spectrogram for ' + os.path.basename(audio_file)))

    plt.title(str(plot_title))
    plt.show()

def spectrogramWindow(audio_file):
    """
    Creates a window to configure the spectrogram with.

    Args:
        audio_file (str): audio file to be passed to the spectrogram function.
    """    
    spectrogramConfigWindow = tk.Toplevel(root)
    spectrogramConfigWindow.iconphoto(False, icon_photo)
    centered_window(spectrogramConfigWindow)
    spectrogramConfigWindow.grid_columnconfigure(0, weight=1)
    spectrogramConfigWindow.grid_rowconfigure(0, weight=1)
    spectrogramConfigWindow.minsize(550, 250)
    spectrogramConfigWindow.title('Configure Spectrogram')
    spectrogramConfigWindowContentFrame = tk.Frame(spectrogramConfigWindow, borderwidth=10, relief='flat')
    spectrogramConfigWindowContentFrame.grid(row=0, column=0)

    spectrogramConfigTitleLabel = tk.Label(spectrogramConfigWindowContentFrame, text='Configure Spectrogram', font=("TkDefaultFont", str(parse_font_dict['size'] + 2), "bold"))
    spectrogramConfigTitleLabel.grid(row=0, column=1)
    spectrogramAudioTitleLabel = tk.Label(spectrogramConfigWindowContentFrame, text='Audio file:')
    spectrogramAudioTitleLabel.grid(row=1, column=0)
    spectrogramSelectedAudioLabel = tk.Label(spectrogramConfigWindowContentFrame, text=shorten_file_name(os.path.basename(audio_file), 24))
    create_tooltip(spectrogramSelectedAudioLabel, text=str(audio_file))
    spectrogramSelectedAudioLabel.grid(row=1, column=2)
    startTimeAudioLabel = tk.Label(spectrogramConfigWindowContentFrame, text='Start time (ms):')
    startTimeAudioLabel.grid(row=2, column=0)
    endTimeAudioLabel = tk.Label(spectrogramConfigWindowContentFrame, text='End time (ms):')
    endTimeAudioLabel.grid(row=3, column=0)
    dynamicRangeMinLabel = tk.Label(spectrogramConfigWindowContentFrame, text='Intensity scale min (dB):')
    dynamicRangeMinLabel.grid(row=4, column=0)
    dynamicRangeMaxLabel = tk.Label(spectrogramConfigWindowContentFrame, text='Intensity scale max (dB):')
    dynamicRangeMaxLabel.grid(row=5, column=0)
    plotTitleLabel = tk.Label(spectrogramConfigWindowContentFrame, text='Plot title:')
    plotTitleLabel.grid(row=6, column=0)

    startTimeStringVar = tk.StringVar()
    endTimeStringVar = tk.StringVar()
    dynamicRangeMinStringVar = tk.StringVar()
    dynamicRangeMaxStringVar = tk.StringVar()
    plotTitleStringVar = tk.StringVar()

    startTimeAudioEntry = tk.Entry(spectrogramConfigWindowContentFrame, textvariable=startTimeStringVar)
    startTimeAudioEntry.grid(row=2, column=2)
    endTimeAudioEntry = tk.Entry(spectrogramConfigWindowContentFrame, textvariable=endTimeStringVar)
    endTimeAudioEntry.grid(row=3, column=2)
    dynamicRangeMinEntry = tk.Entry(spectrogramConfigWindowContentFrame, textvariable=dynamicRangeMinStringVar)
    dynamicRangeMinEntry.grid(row=4, column=2)
    dynamicRangeMaxEntry = tk.Entry(spectrogramConfigWindowContentFrame, textvariable=dynamicRangeMaxStringVar)
    dynamicRangeMaxEntry.grid(row=5, column=2)
    plotTitleEntry = tk.Entry(spectrogramConfigWindowContentFrame, textvariable=plotTitleStringVar)
    plotTitleEntry.grid(row=6, column=2)

    viewSpectrogramButton = tk.Button(spectrogramConfigWindowContentFrame, text='View spectrogram', command=lambda:spectrogram(audio_file, startTimeStringVar, endTimeStringVar, dynamicRangeMinStringVar, dynamicRangeMaxStringVar, plotTitleStringVar))
    viewSpectrogramButton.grid(row=7, column=1)

def errorWindow(error_message='Generic Error Message', title='Error', width=300, height=100, **kwargs):
    """
    Creates an error window.

    Args:
        error_message (str, optional): Error message to be displayed. Defaults to 'Generic Error Message'.
        title (str, optional): Title of window. Defaults to 'Error'.
        width (int, optional): Width of window. Defaults to 300.
        height (int, optional): Height of window. Defaults to 100.
    
    Keyword Arguments:
        tooltip_text (str, optional): If a string is provided, a tooltip will appear with that string when hovering over the error message.

    Returns:
        
    """    
    tooltip_text = kwargs.get('tooltip_text', None)
    errorWindow = tk.Toplevel(root)
    errorWindow.iconphoto(False, icon_photo)
    centered_window(errorWindow)
    errorWindow.title(str(title))
    errorWindow.geometry(str(width)+'x'+str(height))
    errorWindow.minsize(width, height)
    errorMessageLabel = tk.Label(errorWindow, text='\n\nError: ' + str(error_message))
    if tooltip_text:
        create_tooltip(errorMessageLabel, text=str(tooltip_text))
    errorMessageLabel.pack()
    return -1

def messageWindow(message='Generic Message', title='Title', width=300, height=100, **kwargs):
    """
    Creates a message/alert window.

    Args:
        message (str, optional): Message to be displayed. Defaults to 'Generic Message'.
        title (str, optional): Title of window. Defaults to 'Title'.
        width (int, optional): Width of window. Defaults to 300.
        height (int, optional): Height of window. Defaults to 100.
    
    Keyword Arguments:
        tooltip_text (str, optional): If a string is provided, a tooltip will appear with that string when hovering over the message.

    Returns:
    
    """    
    tooltip_text = kwargs.get('tooltip_text', None)
    messageWindow = tk.Toplevel(root)
    messageWindow.iconphoto(False, icon_photo)
    centered_window(messageWindow)
    messageWindow.title(str(title))
    messageWindow.geometry(str(width)+'x'+str(height))
    messageWindow.minsize(width, height)
    messageLabel = tk.Label(messageWindow, text='\n\n' + str(message))
    if tooltip_text:
        create_tooltip(messageLabel, text=str(tooltip_text))
    messageLabel.pack()
    return -1

def callback_url(url):
    webbrowser.open_new(url)

def clearWidgets(frame_to_clear):
    for widget in frame_to_clear.winfo_children():
        widget.destroy()

def createHelpWindow():
    global tutorialWindow
    global tutorialWindowContentFrame
    tutorialWindow = tk.Toplevel(root)
    tutorialWindow.iconphoto(False, icon_photo)
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

    selectSourceFileTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Select Source File (.wav)"\nPresents dialogue box for selecting a source file.\nOnly takes .wav files.\nUsed for convolving with loaded HRTF.\nAlso used for convolving with loaded SOFA file.\n')
    selectSourceFileTutorialLabel.grid(row=1, column=2)
    getSourceFileDataTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Get Source File Data"\nPresents info about source file, including:\n- Sample rate\n- Data dimension (num samples, num channels)\n- Ability to play loaded selected file.\n')
    getSourceFileDataTutorialLabel.grid(row=2, column=2)
    spectrogramTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"View Spectrogram"\nOpens a menu for configuring and\n viewing a spectrogram of the loaded source file.\n')
    spectrogramTutorialLabel.grid(row=3, column=2)
    stereoToMonoTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Source File Stereo -> Mono"\nDownmixes stereo file into mono\nfor convenience with convolving.\n')
    stereoToMonoTutorialLabel.grid(row=4, column=2)

    resampleTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Resample"\nResamples source file to match sample rate of loaded HRTF file.\nResampled source file is held in memory, not exported.\nSource File Stereo -> Mono MUST be pressed first!\n')
    resampleTutorialLabel.grid(row=5, column=1)
    timeDomainConvolveTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Time Domain Convolve"\nTime domain convolves loaded source file with loaded HRTF file.\nSource file should either match HRTF file sample rate,\nor have been resampled with the button above.\nConvolved file is held in memory, not exported.\n')
    timeDomainConvolveTutorialLabel.grid(row=6, column=1)
    exportConvolvedTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Export Convolved"\nExports the time domain convolved file loaded in memory.\nFile naming convention is:\n[Source File Name]-[HRTF File Name]-export.wav\n')
    exportConvolvedTutorialLabel.grid(row=7, column=1)

    nextButton = tk.Button(tutorialWindowContentFrame, text='Next (SOFA Help) ->', command=lambda:sofaHelpPage())
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
    frequencyXLimTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Frequency Range (Hz)"\nConfigurable range for x-axis of .SOFA file plot.\nDefaults to [20, 20000].\n')
    frequencyXLimTutorialLabel.grid(row=4, column=0)
    desiredAzimuthTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Desired azimuth (in deg)"\nEnter azimuth for rendering with source file. Defaults to 0deg.\nSelectable azimuth for viewing .SOFA file plot, and for rendering.\n')
    desiredAzimuthTutorialLabel.grid(row=5, column=0)

    getSOFAFileDimensionsTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Get SOFA File Dimensions"\nPresents info about the SOFA convention dimensions\nwithin the .SOFA file.\n')
    getSOFAFileDimensionsTutorialLabel.grid(row=2, column=2)
    sofaEmitterTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Emitter"\nChoose the emitter to be used when viewing plot data.\nCheck .SOFA file dimensions for emitters.\n')
    sofaEmitterTutorialLabel.grid(row=3, column=2)
    magnitudeYLimTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Magnitude (dB)"\nConfigurable range for y-axis of .SOFA file plot.\nDefaults to [-150, 0].\n')
    magnitudeYLimTutorialLabel.grid(row=4, column=2)
    desiredElevationTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Desired elevation (in deg)"\nEnter elevation for rendering with source file. Defaults to 0deg.\nSelectable elevation for viewing .SOFA file plot, and for rendering.\n')
    desiredElevationTutorialLabel.grid(row=5, column=2)

    viewSOFAFileTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"View SOFA File"\nTakes the above selected values\nand presents a 3D view of the .SOFA file,\n in addition to individual measurements\nfrom the .SOFA file.\n')
    viewSOFAFileTutorialLabel.grid(row=6, column=0)
    saveSOFAFileTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Save all SOFA graphs"\nSaves graphs/plots for source positions,\nhead-related impulse response, and head-related transfer function\nfor the provided azimuth, elevation, emitter, and measurement index to\nthe provided directory.\n')
    saveSOFAFileTutorialLabel.grid(row=6, column=2)
    renderSOFATutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Render Source with SOFA File"\nConvolves the source file with\nthe desired values in the .SOFA file.\n')
    renderSOFATutorialLabel.grid(row=7, column=1)

    prevButton = tk.Button(tutorialWindowContentFrame, text='<- Previous (HRTF Help)', command=lambda:hrtfHelpPage())
    prevButton.grid(row=8, column=0, sticky='W')

    nextButton = tk.Button(tutorialWindowContentFrame, text='Next (General Help) ->', command=lambda:generalHelpPage())
    nextButton.grid(row=8, column=2, sticky='E')

def generalHelpPage():
    clearWidgets(tutorialWindowContentFrame)
    generalTitleTutorialLabel = tk.Label(tutorialWindowContentFrame, text='General Help\n', font=("TkDefaultFont", str(parse_font_dict['size'] + 4), "bold"))
    generalTitleTutorialLabel.grid(row=0, column=1)

    whatIsTitleTutorialLabel = tk.Label(tutorialWindowContentFrame, text='What is this program?', font=("TkDefaultFont", str(parse_font_dict['size'] + 2), "bold"))
    whatIsTitleTutorialLabel.grid(row=1, column=0)
    whatIsDescTutorialLabel = tk.Label(tutorialWindowContentFrame, text='''
    "Garrett's Great HRTF (& SOFA) Functions" is a collected suite
    of functions for analyzing and operating upon files for immersive
    audio applications.
    \nFor HRTFs, these functions include time and frequency domain graphing
    for HRIR/HRTF .wav files, time domain convolution of any .wav audio
    file with the loaded HRIR .wav file, the ability to quickly view
    metadata about loaded HRTF files like data dimensions and sample rate,
    previewing loaded HRIR files, resampling, exporting convolved files, and more.
    \nFor SOFA files, these functions include viewing metadata, viewing
    measurements at configurable points, graphic view of the SOFA file,
    the ability to convolve any .wav audio file with the loaded SOFA file
    at any captured azimuth and elevation within the SOFA file, and more.''')
    whatIsDescTutorialLabel.grid(row=2, column=0)
    howToTitleTutorialLabel = tk.Label(tutorialWindowContentFrame, text='How do I use this program?', font=("TkDefaultFont", str(parse_font_dict['size'] + 2), "bold"))
    howToTitleTutorialLabel.grid(row=1,column=2)
    howToDescTutorialLabel = tk.Label(tutorialWindowContentFrame, text='''
    To analyze, load the file you would like to analyze (HRTF or SOFA).
    Every function should now be available for analysis, aside from the
    functions that are intended for convolution.
    \nTo convolve, load a source file to be convolved. This source file
    will be used regardless of whether you are convolving with a loaded HRIR/HRTF,
    or with a loaded SOFA file.
    To convolve with a loaded HRIR/HRTF, the source file must first be summed
    to mono, then resampled, and then time domain convolved (in that order), before
    exporting.
    To convolve with a loaded SOFA file, choose the measurement index and emitter,
    then the desired azimuth and elevation that the source file should be convolved
    with. Frequency range and magnitude are only for configuring visualization, and do not
    impact convolution. If azimuth and elevation are not configured, they will
    both default to 0. If measurement index and emitter are not configured, they will
    default to 0 and 1 respectively. 
    ''')
    howToDescTutorialLabel.grid(row=2, column=2)
    
    commonHelpTitleTutorialLabel = tk.Label(tutorialWindowContentFrame, text='Tips/Tricks', font=("TkDefaultFont", str(parse_font_dict['size'] + 2), "bold"))
    commonHelpTitleTutorialLabel.grid(row=3,column=0)
    commonHelpDescTutorialLabel = tk.Label(tutorialWindowContentFrame, text='''
    * 'Source File Stereo -> Mono' must ALWAYS be pressed before using HRTF functions!
    * You can use 'Tab' to select a subsequent text box.
    * The default values for SOFA functions are as follow:
        - Measurement Index: 0
        - Emitter: 1
        - Frequency Range (Hz): 20, 20000
        - Magnitude (dB): -150, 0
        - Azimuth (deg): 0
        - Elevation (deg): 0
    ''', justify='left')
    commonHelpDescTutorialLabel.grid(row=4, column=0, rowspan=8)
    
    feedbackTitleTutorialLabel = tk.Label(tutorialWindowContentFrame, text='Contact/Feedback', font=("TkDefaultFont", str(parse_font_dict['size'] + 2), "bold"))
    feedbackTitleTutorialLabel.grid(row=3,column=2)
    feedbackDescTutorialLabel = tk.Label(tutorialWindowContentFrame, text='https://hytheaway.github.io/contact.html', fg="blue", cursor='hand2')
    feedbackDescTutorialLabel.grid(row=4, column=2)
    feedbackDescTutorialLabel.bind("<Button-1>", lambda e: callback_url("https://hytheaway.github.io/contact.html"))
    feedbackDesc2TutorialLabel = tk.Label(tutorialWindowContentFrame, text='https://github.com/hytheaway', fg="blue", cursor='hand2')
    feedbackDesc2TutorialLabel.grid(row=5, column=2)
    feedbackDesc2TutorialLabel.bind("<Button-1>", lambda e: callback_url("https://github.com/hytheaway"))
    feedbackDesc3TutorialLabel = tk.Label(tutorialWindowContentFrame, text='hytheaway@gmail.com', fg="blue", cursor='hand2')
    feedbackDesc3TutorialLabel.grid(row=6, column=2)
    feedbackDesc3TutorialLabel.bind("<Button-1>", lambda e: callback_url("mailto:hytheaway@gmail.com"))

    prevButton = tk.Button(tutorialWindowContentFrame, text='<- Previous (SOFA Help)', command=lambda:sofaHelpPage())
    prevButton.grid(row=20, column=0, sticky='W')

# matplotlib.use('QtAgg')

root = tk.Tk()
root.minsize(565, 890)
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
centered_window(root)
icon_photo = tk.PhotoImage(file='assets/happyday.png')
root.iconphoto(False, icon_photo)
root.title("GGH&SF")
default_font = font.nametofont("TkDefaultFont")
parse_font_dict = default_font.actual()

icon_label = tk.Label(root, image=icon_photo)
icon_label.place(x=0, y=0, width=120, height=120, relwidth=0.5, relheight=0.5)

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
selectSourceFileButton = tk.Button(sourceFrame, text='Select source file (.wav)', command=lambda:selectSourceFile())
selectSourceFileLabel = tk.Label(sourceFrame, text='Source file:\n', wraplength=120)
selectSourceFileButton.grid(row=0, column=2)
selectSourceFileLabel.grid(row=1, column=2)
getSourceFileDataButton = tk.Button(sourceFrame, text='Get Source File Data', state='disabled', command=lambda:getSourceFileData())
getSourceFileDataButton.grid(row=2, column=2)
spectrogramButton = tk.Button(sourceFrame, text='View Spectrogram', state='disabled', command=lambda:spectrogramWindow(source_file))
spectrogramButton.grid(row=3, column=2)
stereoToMonoButton = tk.Button(sourceFrame, text='Source File Stereo -> Mono', state='disabled', command=lambda:stereoToMono())
stereoToMonoButton.grid(row=4, column=2)

hrtfOperationsFrame = tk.Frame(topSectionFrame, borderwidth=10, relief='flat')
hrtfOperationsFrame.grid(row=1, column=1)

resampleButton = tk.Button(hrtfOperationsFrame, text='Resample', state='disabled', command=lambda:fs_resample(sig_mono, fs_s, HRIR, fs_H))
resampleButton.grid(row=1, column=1)
timeDomainConvolveButton = tk.Button(hrtfOperationsFrame, text='Time Domain Convolve', state='disabled', command=lambda:timeDomainConvolve())
timeDomainConvolveButton.grid(row=2, column=1)
exportConvolvedButton = tk.Button(hrtfOperationsFrame, text='Export Convolved', state='disabled', command=lambda:exportConvolved())
exportConvolvedButton.grid(row=3, column=1)

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

sofaMeasurementStringVar = tk.StringVar()
sofaEmitterStringVar = tk.StringVar()
freqXLimStringVar = tk.StringVar()
magYLimStringVar = tk.StringVar()
azimuthStringVar = tk.StringVar()
elevationStringVar = tk.StringVar()

bottomLeftFrame = tk.Frame(bottomSectionFrame, borderwidth=10, relief='flat')
bottomLeftFrame.grid(row=2, column=0)
bottomRightFrame = tk.Frame(bottomSectionFrame, borderwidth=10, relief='flat')
bottomRightFrame.grid(row=2, column=2)

getSOFAFileMetadataButton = tk.Button(bottomLeftFrame, text='Get SOFA File Metadata', state='disabled', command=lambda:getSOFAFileMetadata())
getSOFAFileMetadataButton.grid(row=0, column=0)
getSOFAFileDimensionsButton = tk.Button(bottomRightFrame, text='Get SOFA File Dimensions', state='disabled', command=lambda:getSOFAFileDimensions())
getSOFAFileDimensionsButton.grid(row=0, column=0)

sofaMeasurementTextBox = tk.Entry(bottomLeftFrame, state='disabled', width=5, textvariable=sofaMeasurementStringVar)
sofaMeasurementTextBox.grid(row=1, column=0)
sofaMeasurementLabel = tk.Label(bottomLeftFrame, text='Measurement Index\n(default: 0)\n')
sofaMeasurementLabel.grid(row=2, column=0)
sofaEmitterTextBox = tk.Entry(bottomRightFrame, state='disabled', width=5, textvariable=sofaEmitterStringVar)
sofaEmitterTextBox.grid(row=1, column=0)
sofaEmitterLabel = tk.Label(bottomRightFrame, text='Emitter\n(default: 1)\n')
sofaEmitterLabel.grid(row=2, column=0)

frequencyXLimTextBox = tk.Entry(bottomLeftFrame, state='disabled', width=15, textvariable=freqXLimStringVar)
frequencyXLimTextBox.grid(row=3, column=0)
frequencyXLimLabel = tk.Label(bottomLeftFrame, text='Frequency Range (Hz)\n[start, end]')
frequencyXLimLabel.grid(row=4, column=0)
magnitudeYLimTextBox = tk.Entry(bottomRightFrame, state='disabled', width=15, textvariable=magYLimStringVar)
magnitudeYLimTextBox.grid(row=3, column=0)
magnitudeYLimLabel = tk.Label(bottomRightFrame, text='Magnitude (dB)\n[start, end]')
magnitudeYLimLabel.grid(row=4, column=0)

azimuthTextBox = tk.Entry(bottomLeftFrame, state='disabled', width=5, textvariable=azimuthStringVar)
azimuthTextBox.grid(row=5, column=0)
azimuthLabel = tk.Label(bottomLeftFrame, text='Desired azimuth (in deg)')
azimuthLabel.grid(row=6, column=0)
elevationTextBox = tk.Entry(bottomRightFrame, state='disabled', width=5, textvariable=elevationStringVar)
elevationTextBox.grid(row=5, column=0)
elevationLabel = tk.Label(bottomRightFrame, text='Desired elevation (in deg)')
elevationLabel.grid(row=6, column=0)

sofaViewButton = tk.Button(bottomSectionFrame, text='View SOFA File', state='disabled', command=lambda:displaySOFAFile(freqXLimStringVar.get(), magYLimStringVar.get(), sofaMeasurementStringVar.get(), sofaEmitterStringVar.get()))
sofaViewButton.grid(row=3, column=0, columnspan=2)
sofaSaveButton = tk.Button(bottomSectionFrame, text='Save all SOFA graphs', state='disabled', command=lambda:saveSOFAFile(freqXLimStringVar.get(), magYLimStringVar.get(), sofaMeasurementStringVar.get(), sofaEmitterStringVar.get()))
sofaSaveButton.grid(row=3, column=1, columnspan=2)
sofaDisplayButton = tk.Button(bottomSectionFrame, text='Render Source with SOFA file', state='disabled', command=lambda:manualSOFADisplay(azimuthStringVar.get(), elevationStringVar.get(), source_file))
sofaDisplayButton.grid(row=4, column=0, columnspan=3)

tutorialButton = tk.Button(rootFrame, text='Help', command=lambda:createHelpWindow())
tutorialButton.grid(row=5, column=0, sticky='W')

quitButton = tk.Button(rootFrame, text='Quit', command=lambda:root.destroy())
quitButton.grid(row=5, column=2, sticky='E')

# prevents the window from appearing at the bottom of the stack
root.attributes('-topmost', True)
root.update()
root.attributes('-topmost', False)
root.focus_force()

root.mainloop()