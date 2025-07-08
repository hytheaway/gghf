# much of this is sourced from andrea genovese
# please find him here: https://andreagenovese.com/
# and his wonderful tutorial about hrtf processing here: https://www.youtube.com/watch?v=a4mpK_2koR4

# also please keep in mind that this isn't supposed to be "efficient" or "clean" or "lightweight".
# this is meant to be the most brute force way to do all my hrtf processing in one file with one interface. 

import os # <- reading files from disk, adapting to differing os directory path conventions
import tempfile # <- adapting to differing os temp file locations

os.environ['LIBROSA_CACHE_DIR'] = str(tempfile.gettempdir()) # <- must be called before importing librosa
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide' # <- gets rid of pygame welcome message, which clutters up cli

import numpy as np # <- matrix calc & more (but mostly matrix calc)
import matplotlib.pyplot as plt # <- data visualization 
import soundfile as sf # <- read audio into ndarray
import sofa # <- read SOFA HRTFs
import librosa # <- resample function
from scipy import signal # <- fast convolution function
from scipy.io import wavfile # <- used for spectrogram
import pygame # <- for playing audio files directly
import tkinter as tk # <- reliable, if clunky, gui
from tkinter import ttk # <- necessary so buttons don't turn invisible with dark mode enabled
from tkinter import filedialog # <- gui file selection from disk
from tkinter import font # <- ensures fonts are system-compatible
import webbrowser # <- help page links
import sv_ttk # <- handles ttk
import darkdetect # <- detects os light/dark mode

librosa.cache.clear(warn=False)

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
        tooltip_label = tk.Label(tw, text=self.text, justify='left', background='#ffffe0', foreground='black', relief='solid', borderwidth=0, font="TkDefaultFont 10 normal")
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

def errorWindow(error_message:str='Generic Error Message', title:str='Error', width:int=300, height:int=100, **kwargs):
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
    errorMessageLabel = ttk.Label(errorWindow, text='\nError: ' + str(error_message) + '\n', justify='center') 
    if tooltip_text:
        create_tooltip(errorMessageLabel, text=str(tooltip_text))
    errorMessageLabel.pack()
    errorConfirmButton = ttk.Button(errorWindow, text='OK', command=lambda:errorWindow.destroy())
    errorConfirmButton.pack()
    errorWindow.focus_force()
    return -1

def messageWindow(message:str='Generic Message', title:str='Title', width:int=300, height:int=100, **kwargs):
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
    messageLabel = ttk.Label(messageWindow, text='\n' + str(message) + '\n', justify='center')
    if tooltip_text:
        create_tooltip(messageLabel, text=str(tooltip_text))
    messageLabel.pack()
    messageConfirmButton = ttk.Button(messageWindow, text='OK', command=lambda:messageWindow.destroy())
    messageConfirmButton.pack()
    messageWindow.focus_force()
    return -1

def shorten_file_name(old_filename:str, num_shown_char:int):
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

def playAudio(path_to_audio_file:str):
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
    root.focus_force()
    try:
        if hrtf_file:
            [HRIR,fs_H] = sf.read(hrtf_file)
    except RuntimeError:
        errorWindow('\nError loading file:\n\n' + str(os.path.basename(hrtf_file)) + '\n\nRuntime error.\nDoes this file exist on the local drive?', title='Error', width=300, height=200, tooltip_text=hrtf_file)
        return
    else:
        if hrtf_file:
            hrtf_file_print = hrtf_file.split('/')
            selectHRTFFileLabel.config(text='HRTF file:\n' + shorten_file_name(os.path.basename(hrtf_file), 13))
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
    root.focus_force()
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
        selectSourceFileLabel.config(text='Source file:\n' + shorten_file_name(os.path.basename(source_file), 13))
        create_tooltip(selectSourceFileLabel, text=str(source_file))
        getSourceFileDataButton.config(state='active')
        stereoToMonoButton.config(state='active')
        spectrogramButton.config(state='active')
        resampleButton.config(state='disabled')
        [sig,fs_s] = sf.read(source_file)

def getHRTFFileData(in_hrtf_file:str, in_HRIR:np.ndarray):
    """
    For a given HRTF file, create a window, read and display the sample rate and data dimensions, and create buttons to play & pause the HRTF, and close the window.

    Args:
        in_hrtf_file (str): Path to HRTF file (.wav).
        in_HRIR (np.ndarray): HRIR object pulled from in_hrtf_file.
    """
    if len(in_HRIR.shape) < 2:
        errorHrtfFileData = [str('Selected file: ' + os.path.basename(in_hrtf_file)), str('Sample rate: ' + str(fs_H)), str('Data dimensions: ' + str(in_HRIR.shape))]
        errorWindow(error_message='Selected file only has one dimension (mono channel).\nAre you sure this is an HRTF/HRIR?\nHover for more info.', width=400, height=125, tooltip_text='\n'.join(map(str, errorHrtfFileData)))
        return -1
    else:
        hrtfFileDataWindow = tk.Toplevel(root)
        hrtfFileDataWindow.iconphoto(False, icon_photo)
        centered_window(hrtfFileDataWindow)
        hrtfFileDataWindow.minsize(300, 120)
        hrtfFileDataWindow.title('HRTF File Data')
        windowTitleHRTFData = ttk.Label(hrtfFileDataWindow, text='\n' + os.path.basename(in_hrtf_file) + '\n', justify='center')
        windowTitleHRTFData.pack()
        create_tooltip(windowTitleHRTFData, in_hrtf_file)
        hrtfSampleRateLabel = ttk.Label(hrtfFileDataWindow, text='Sample rate: ' + str(fs_H), justify='center')
        hrtfSampleRateLabel.pack()
        hrtfDataDimLabel = ttk.Label(hrtfFileDataWindow, text='Data dimensions: ' + str(in_HRIR.shape) + '\n', justify='center')
        hrtfDataDimLabel.pack()
        hrtfPlayFileButton = ttk.Button(hrtfFileDataWindow, text='Play HRTF', command=lambda:playAudio(in_hrtf_file))
        hrtfPlayFileButton.pack()
        hrtfPauseFileButton = ttk.Button(hrtfFileDataWindow, text='Pause HRTF', command=lambda:pygame.mixer.music.pause())
        hrtfPauseFileButton.pack()
        hrtfFileDataCloseWindowButton = ttk.Button(hrtfFileDataWindow, text='Close', command=lambda:stopAudioAndCloseWindow(hrtfFileDataWindow))
        hrtfFileDataCloseWindowButton.pack()

def getSourceFileData(in_source_file:str):
    """
    For a given audio file, create a window, read and display the sample rate and data dimensions, and create buttons to play & pause the audio, and close the window.

    Args:
        in_source_file (str): Path to audio file (.wav).
    """    
    sourceFileDataWindow = tk.Toplevel(root)
    sourceFileDataWindow.iconphoto(False, icon_photo)
    centered_window(sourceFileDataWindow)
    sourceFileDataWindow.minsize(300, 120)
    sourceFileDataWindow.title('Source File Data')
    windowTitleSourceData = ttk.Label(sourceFileDataWindow, text='\n' + os.path.basename(in_source_file) + '\n', justify='center')
    windowTitleSourceData.pack()
    create_tooltip(windowTitleSourceData, in_source_file)
    sourceSampleRateLabel = ttk.Label(sourceFileDataWindow, text='Sample rate: ' + str(fs_s), justify='center')
    sourceSampleRateLabel.pack()
    sourceDataDimLabel = ttk.Label(sourceFileDataWindow, text='Data dimensions: ' + str(sig.shape) + '\n', justify='center')
    sourceDataDimLabel.pack()
    sourcePlayFileButton = ttk.Button(sourceFileDataWindow, text='Play Source File', command=lambda:playAudio(in_source_file))
    sourcePlayFileButton.pack()
    sourcePauseFileButton = ttk.Button(sourceFileDataWindow, text='Pause Source File', command=lambda:pygame.mixer.music.pause())
    sourcePauseFileButton.pack()
    sourceFileDataCloseWindowButton = ttk.Button(sourceFileDataWindow, text='Close', command=lambda:(stopAudioAndCloseWindow(sourceFileDataWindow)))
    sourceFileDataCloseWindowButton.pack()

def stopAudioAndCloseWindow(window_to_close):
    # cannot believe i'm actually making this function
    try:
        pygame.mixer.music.pause()
    except pygame.error:
        pass
    finally:
        window_to_close.destroy()

def timeDomainVisualHRTF(in_hrtf_file:str, in_HRIR:np.ndarray):
    """
    For a given HRTF, plot the time domain visualization.

    Args:
        in_hrtf_file (str): Path to HRTF file (.wav). 
        in_HRIR (np.ndarray): HRIR object pulled from in_hrtf_file.
    """    
    plt.figure(num=str('Time Domain for ' + os.path.basename(in_hrtf_file)))
    plt.plot(in_HRIR[:,0]) # left channel data
    plt.plot(in_HRIR[:,1]) # right channel data
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('HRIR at angle: ' + os.path.basename(in_hrtf_file))
    plt.legend(['Left','Right'])
    plt.show()

def freqDomainVisualHRTF(in_hrtf_file:str):
    """
    For a given HRTF, plot the frequency domain visualization.

    Args:
        in_hrtf_file (str): Path to HRTF file (.wav).
    """    
    nfft = len(HRIR)*8
    HRTF = np.fft.fft(HRIR,n=nfft, axis=0)
    HRTF_mag = (2/nfft)*np.abs(HRTF[0:int(len(HRTF)/2)+1,:])
    HRTF_mag_dB = 20*np.log10(HRTF_mag)

    f_axis = np.linspace(0,fs_H/2,len(HRTF_mag_dB))
    plt.figure(num=str('Frequency Domain for ' + os.path.basename(in_hrtf_file)))
    plt.semilogx(f_axis, HRTF_mag_dB)
    plt.grid()
    plt.grid(which='minor', color="0.9")
    plt.title('HRTF at angle: ' + os.path.basename(in_hrtf_file))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend(['Left','Right'])
    plt.show()

def stereoToMono(in_sig:np.ndarray):
    """
    Given a signal, evaluate if it is stereo or mono, and if stereo, convert to mono. Creates a information window with relevant information to the process.

    Args:
        in_sig (numpy.ndarray): Signal to be converted to mono.
    """    
    global sig_mono
    if len(in_sig.shape) > 1:
        if in_sig.shape[1] > 1:
            sig_mono = np.mean(in_sig, axis=1)
        else:
            sig_mono = in_sig
        messageWindow(message=('New source data dimensions:\n' + str(sig_mono.shape)), title='Stereo -> Mono', width=350, height=110)
    else:
        sig_mono = in_sig
        messageWindow(message='Source file is already mono.', title='Stereo -> Mono', width=350)
    if getHRTFFileDataButton['state'] == tk.ACTIVE:
        resampleButton.config(state='active')
    
def fs_resample(s1:np.ndarray, f1:int, s2:np.ndarray, f2:int):
    """
    For two signals that have differing sample rates, resample the lower to meet the higher.
    
    Args:
        s1 (numpy.ndarray): First signal.
        f1 (int): Sampling rate of s1.
        s2 (numpy.ndarray): Second signal.
        f2 (int): Sampling rate of s2.
    
    Returns:
        s1 (numpy.ndarray): Resampled signal 1.
        f1 (int): signal 1's resampled sampling rate.
        s2 (numpy.ndarray): resampled signal 2.
        f2 (int): signal 2's resampled sampling rate.
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
    
    messageWindow(message=('Resampled at: ' + str(fmax) + 'Hz\n\n' + 'Signal/source dimensions: ' + str(s1.shape) + '\n' + 'HRIR Dimensions: ' + str(s2.shape)), title='Resample', width=250, height=150)
    
    timeDomainConvolveButton.config(state='active')
    return s1, f1, s2, f2

def timeDomainConvolve(in_sig_mono:np.ndarray, in_HRIR:np.ndarray):
    """
    For a given mono signal, time domain convolve it by the HRIR.

    Args:
        in_sig_mono (np.ndarray): Mono signal in np.ndarray format.
        in_HRIR (np.ndarray): HRIR signal in np.ndarray format.
    """    
    global Bin_Mix
    s_L = np.convolve(in_sig_mono,in_HRIR[:,0])
    s_R = np.convolve(in_sig_mono,in_HRIR[:,1])
    
    Bin_Mix = np.vstack([s_L,s_R]).transpose()
    
    messageWindow(message=('New data dimensions: ' + str(Bin_Mix.shape)), title='Time Domain Convolve', width=250)
    
    exportConvolvedButton.config(state='active')

# freq domain convolution goes here at some point

def exportConvolved(in_Bin_Mix:np.ndarray, in_fs_s:int, in_source_file:str, in_hrtf_file:str):
    """
    For a given frequency bin and sampling rate, export a file with the naming convention of [in_source_file]-[in_hrtf_file]-export.wav.

    Args:
        in_Bin_Mix (np.ndarray): ndarray of frequencies to be exported.
        in_fs_s (int): Sampling rate to export in_Bin_Mix to.
        in_source_file (str): First half of the output name (usually the source file's name).
        in_hrtf_file (str): Second half of the output name (usually the HRTF file's name).

    Returns:

    """    
    source_file_location = os.path.join(*source_file_print[:-1])
    export_directory = filedialog.askdirectory(title='Select Save Directory', initialdir=source_file_location)
    convolved_filename = str(os.path.basename(in_source_file)[:-4]) + '-' + str(os.path.basename(in_hrtf_file)[:-4]) + '-export.wav'
    if export_directory:
        export_filename = os.path.join(str(export_directory), str(convolved_filename))
        sf.write(export_filename, in_Bin_Mix, in_fs_s)
    if not export_directory:
        return -1

    if os.path.exists(export_filename):
        messageWindow(message=('File successfully exported as:\n' + convolved_filename), title='Export Successful', width=400, tooltip_text=str(export_filename))
    else:
        errorWindow(error_message='Export Failed')

def selectSOFAFile():
    global sofa_file_print
    global sofa_file_path_list
    global sofa_mode_selection
    
    sofa_file_path_list = filedialog.askopenfilenames(filetypes=[('.SOFA files', '.sofa')])
    root.focus_force()
    if len(sofa_file_path_list) > 1:
        sofa_mode_selection = 1
        for file in sofa_file_path_list:
            if file:
                try:
                    metadata_test = sofa.Database.open(file).Metadata.list_attributes()
                except OSError:
                    errorWindow('\nError loading file:\n\n' + str(os.path.basename(file)) + '\n\nOS error.\nDoes this file exist on the local drive?\n\nAlternatively, does this SOFA file\ncontain correct metadata?', title='Error', width=300, height=250, tooltip_text=file)
                    return
                else:
                    continue
        selectSOFAFileLabel.config(text='SOFA file:\nHover to see selected files.')
        create_tooltip(selectSOFAFileLabel, ('NOTE: Graph viewing only - Rendering disabled when multiple SOFA files are selected.\n' + '\n'.join(map(str, sofa_file_path_list))))
        getSOFAFileMetadataButton.config(state='disabled')
        getSOFAFileDimensionsButton.config(state='disabled')
        azimuthTextBox.config(state='disabled')
        elevationTextBox.config(state='disabled')
        sofaRenderButton.config(state='disabled')
        sofaViewButton.config(text='View SOFA HRTF')
        sofaSaveButton.config(text='Save SOFA HRTF')
        
    if len(sofa_file_path_list) == 1:
        sofa_mode_selection = 0
        try:
            if sofa_file_path_list[0]:
                metadata_test = sofa.Database.open(sofa_file_path_list[0]).Metadata.list_attributes()
        except OSError:
            errorWindow('\nError loading file:\n\n' + str(os.path.basename(sofa_file_path_list[0])) + '\n\nOS error.\nDoes this file exist on the local drive?\n\nAlternatively, does this SOFA file\ncontain correct metadata?', title='Error', width=300, height=250, tooltip_text=sofa_file_path_list[0])
            return
        else:
            if sofa_file_path_list[0]:
                sofa_file_print = sofa_file_path_list[0].split('/')
                selectSOFAFileLabel.config(text='SOFA file:\n' + shorten_file_name(os.path.basename(sofa_file_path_list[0]), 20))
                create_tooltip(selectSOFAFileLabel, text=str(sofa_file_path_list[0]))
                getSOFAFileMetadataButton.config(state='active')
                getSOFAFileDimensionsButton.config(state='active')
                azimuthTextBox.config(state='normal')
                elevationTextBox.config(state='normal')
                sofaRenderButton.config(state='active')
                sofaViewButton.config(text='View SOFA File')
                sofaSaveButton.config(text='Save all SOFA graphs')
            else:
                return

    if sofa_file_path_list:
        sofaViewButton.config(state='active')
        sofaSaveButton.config(state='active')
        sofaMeasurementTextBox.config(state='normal')
        sofaEmitterTextBox.config(state='normal')
        frequencyXLimTextBox.config(state='normal')
        magnitudeYLimTextBox.config(state='normal')

def getSOFAFileMetadata(in_sofa_file:str):
    """
    Retrieve, parse, and display in a new window the metadata of a given SOFA file.

    Args:
        in_sofa_file (str): Path to SOFA file.
    """    
    sofaMetadataWindow = tk.Toplevel(root)
    sofaMetadataWindow.iconphoto(False, icon_photo)
    centered_window(sofaMetadataWindow)
    sofaMetadataWindow.geometry('400x400')
    sofaMetadataWindow.title('SOFA File Metadata')
    v = tk.Scrollbar(sofaMetadataWindow, orient='vertical')
    v.pack(side='right', fill='y')
    myString = ''
    for attr in sofa.Database.open(in_sofa_file).Metadata.list_attributes():
        myString = myString + ("{0}: {1}".format(attr, sofa.Database.open(in_sofa_file).Metadata.get_attribute(attr))) + '\n'
    windowSOFAMetadataLabel = tk.Text(sofaMetadataWindow, width=100, height=100, wrap='word', yscrollcommand=v.set)
    windowSOFAMetadataLabel.insert('end', str(myString))
    windowSOFAMetadataLabel.pack()
    
    v.config(command=windowSOFAMetadataLabel.yview)

def getSOFAFileDimensions(in_sofa_file:str):
    """
    Retrieve, parse, and display in a new window the dimensions of a given SOFA file.

    Args:
        in_sofa_file (str): Path to SOFA file.
    """    
    sofaDimensionsWindow = tk.Toplevel(root)
    sofaDimensionsWindow.iconphoto(False, icon_photo)
    centered_window(sofaDimensionsWindow)
    sofaDimensionsWindow.geometry('600x400')
    sofaDimensionsWindow.title('SOFA File Dimensions')
    v = tk.Scrollbar(sofaDimensionsWindow, orient='vertical')
    v.pack(side='right', fill='y')
    definitionsLabel = tk.Label(sofaDimensionsWindow, text='C = Size of coordinate dimension (always three).\n\nI = Single dimension (always one).\n\nM = Number of measurements.\n\nR = Number of receivers or SH coefficients (depending on ReceiverPosition_Type).\n\nE = Number of emitters or SH coefficients (depending on EmitterPosition_Type).\n\nN = Number of samples, frequencies, SOS coefficients (depending on self.GLOBAL_DataType).')
    definitionsLabel.pack()
    myString = ''
    for dimen in sofa.Database.open(in_sofa_file).Dimensions.list_dimensions():
        myString = myString + ("{0}: {1}".format(dimen, sofa.Database.open(in_sofa_file).Dimensions.get_dimension(dimen))) + '\n'
    windowSOFADimensionsLabel = tk.Text(sofaDimensionsWindow, width=10, height=10, wrap='word', yscrollcommand=v.set)
    windowSOFADimensionsLabel.insert('end', str(myString))
    windowSOFADimensionsLabel.pack()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def sanitizeBounds(bounds:str):
    """
    Sanitizes and splits bounds from the user-input frequency and magnitude ranges to account for whitespace, brackets, and commas.
    
    Args:
        bounds (str): Lower and upper bound to be sanitized and split.
    
    Returns:
        str: lower_limit, upper_limit
    """    
    bounds = str(bounds)
    for char in bounds:
        if char == '[' or char == ']' or char == ' ':
            bounds = bounds.replace(char, '')
    for i in range(0, len(bounds)):
        if bounds[i] == ',':
            lower_limit = bounds[0:i]
            upper_limit = bounds[i+1:]
    
    return lower_limit, upper_limit

def plot_coordinates(in_sofa_file:str, fig:plt.Figure):
    """
    Plots source coordinate positions of a SOFA file.

    Args:
        in_sofa_file (str): Path to SOFA file to be plotted.
        fig (matplotlib.figure.Figure): Figure for the data to be plotted on. 

    Returns:
        mpl_toolkits.mplot3d.art3d.Line3DCollection: 3D plot data
    """    
    SOFA_HRTF = sofa.Database.open(in_sofa_file)
    x0 = SOFA_HRTF.Source.Position.get_values(system='cartesian')
    n0 = x0
    ax = fig.add_subplot(111, projection='3d')
    q = ax.quiver(x0[:, 0], x0[:, 1], x0[:, 2], 
                  n0[:, 0], n0[:, 1], n0[:, 2], 
                  length=0.1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Source positions for: ' + str(os.path.basename(in_sofa_file)))
    return q

def computeHRIR(in_sofa_file:str, measurement:int):
    """
    Computes the HRIR for a given SOFA file at the measurement and emitter given. Returns t (time, x-axis), receiver_dimensions, and SOFA_HRTF.Data.IR

    Args:
        in_sofa_file (str): Path to SOFA file
        measurement (int): Measurement value to be operated upon.

    Returns:
        ndarray: t (time, to be used on the x-axis)
        int: receiver dimension
        sofa.access.variables.Variable: IR from the SOFA file to be plotted against t
    """    
    SOFA_HRTF = sofa.Database.open(in_sofa_file)
    
    receiver_dimensions = SOFA_HRTF.Dimensions.R
    t = np.arange(0, SOFA_HRTF.Dimensions.N)*SOFA_HRTF.Data.SamplingRate.get_values(indices={"M":measurement})
    
    return t, receiver_dimensions, SOFA_HRTF.Data.IR

def computeHRTF(in_sofa_file:str, measurement:int, emitter:int):
    """
    Computes the HRTF for a given SOFA file at the measurement and emitter given. Returns f_axis (frequency in Hz, usually used on x-axis) and HRTF_mag_dB (magnitude in dB, usually used on y-axis)

    Args:
        in_sofa_file (str): Path to SOFA file
        measurement (int): Measurement value to be operated upon.
        emitter (int): Emitter to be operated upon.

    Returns:
        ndarray: Frequency axis, usually used on x-axis
        ndarray: Magnitude in dB, usually used on y-axis
    """    
    SOFA_HRTF = sofa.Database.open(in_sofa_file)
    receiver_legend = []
    
    for receiver in np.arange(SOFA_HRTF.Dimensions.R):
        receiver_legend.append("Receiver {0}".format(receiver))
    
    nfft = len(SOFA_HRTF.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))*8
    HRTF = np.fft.fft(SOFA_HRTF.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}), n=nfft, axis=0)
    HRTF_mag = (2/nfft)*np.abs(HRTF[0:int(len(HRTF)/2)+1])
    HRTF_mag_dB = 20*np.log10(HRTF_mag)
    f_axis = np.linspace(0, (SOFA_HRTF.Data.SamplingRate.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))/2, len(HRTF_mag_dB))
    
    return f_axis, HRTF_mag_dB, receiver_legend

def plotHRIR(in_sofa_file, legend:list, measurement:int, emitter:int):
    """
    Plots a head-related impulse response graph for a given .sofa file with a given legend, measurement index, and emitter.

    Args:
        in_sofa_file (str): Path to sofa file.
        legend (list): Legend to populate with receiver dimensions (e.g., Left and Right).
        measurement (int): Measurement index to plot.
        emitter (int): Emitter to plot.
    """    
    plt.figure(figsize=(15, 5), num=('Head-Related Impulse Response at M={0} E={1} for '.format(measurement, emitter) + os.path.basename(in_sofa_file)))
    t, receiver_dimensions, ir = computeHRIR(in_sofa_file, measurement)
    for receiver in np.arange(receiver_dimensions):
        plt.plot(t, ir.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))
        legend.append('Receiver {0}'.format(receiver))
    plt.title('{0}: HRIR at M={1} for emitter {2}'.format(os.path.basename(in_sofa_file), measurement, emitter))
    plt.legend(legend)
    plt.xlabel('$t$ in s')
    plt.ylabel(r'$h(t)$')
    plt.grid()
    
    return

def plotHRTF(in_sofa_file, legend:list, xlim:str, ylim:str, measurement:int, emitter:int):
    """
    Plots a head-related transfer function graph for a given .sofa file with a given legend, x-axis bounds, y-axis bounds, measurement index, and emitter. If multiple SOFA files are selected, they will all be plotted on the same graph.

    Args:
        in_sofa_file (str): Path to sofa file.
        legend (list): Legend to populate.
        xlim (str): Bounds for the x-axis, should be passed in the format [lower, upper] (e.g., [20, 20000]).
        ylim (str): Bounds for the y-axis, should be passed in the format [lower, upper] (e.g., [-150, 0]).
        measurement (int): Measurement index to plot.
        emitter (int): Emitter to plot.
    """    
    xlim_start, xlim_end = sanitizeBounds(xlim)
    ylim_start, ylim_end = sanitizeBounds(ylim)
    
    if sofa_mode_selection == 0:
        plt.figure(figsize=(15, 5), num=('Head-Related Transfer Function at M={0} E={1} for '.format(measurement, emitter) + os.path.basename(in_sofa_file)))
        f_axis, HRTF_mag_dB, legend = computeHRTF(in_sofa_file, measurement, emitter)
        plt.semilogx(f_axis, HRTF_mag_dB)
        plt.title('{0}: HRTF at M={1} for emitter {2}'.format(os.path.basename(in_sofa_file), measurement, emitter))
    
    if sofa_mode_selection == 1:
        in_sofa_files_list = in_sofa_file
        in_sofa_files_list = ', '.join(in_sofa_files_list)
        in_sofa_files_list = in_sofa_files_list.split(",")
        in_sofa_files_list = [file.strip(' ') for file in in_sofa_files_list]        
        plt.figure(figsize=(15, 5), num=str('Left-Channel Head-Related Transfer Function Comparison'))
        for i in in_sofa_files_list:
            legend.append(os.path.basename(i))
            f_axis, HRTF_mag_dB, receiver_legend = computeHRTF(i, measurement, emitter)
            plt.semilogx(f_axis, HRTF_mag_dB)
        plt.title('Left-Channel HRTF Comparison at M={0} for emitter {1}'.format(measurement, emitter))
    
    ax = plt.gca()
    ax.set_xlim([int(xlim_start), int(xlim_end)]) # Bound the x-axis
    ax.set_ylim([int(ylim_start), int(ylim_end)]) # Bound the y-axis
    plt.grid() # Horizontal grid line
    plt.grid(which='minor', color="0.9") # Vertical grid lines
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend(legend)
    
    return

def viewSOFAGraphs(in_sofa_file, xlim:str, ylim:str, measurement:int=0, emitter:int=1):
    """
    Calls functions to plot source positions, HRIR, and HRTF for a given SOFA file, and displays them. Provides default values if they aren't given.

    Args:
        in_sofa_file (str): Path to sofa file to graph.
        xlim (str): Bounds for the x-axis, should be passed in the format [lower, upper] (e.g., [20, 20000]).
        ylim (str): Bounds for the y-axis, should be passed in the format [lower, upper] (e.g., [-150, 0]).
        measurement (int, optional): Measurement index to plot. Defaults to 0.
        emitter (int, optional): Emitter to plot. Defaults to 1.
    """
    
    if not xlim:
        xlim = '20, 20000'
    if not ylim:
        ylim = '-150, 0'
    if not measurement:
        measurement = 0
    if not emitter:
        emitter = 1
    
    legend = []
    
    if sofa_mode_selection == 0:
        in_sofa_file = in_sofa_file[0]

        # plot source coordinates
        sofa_pos_fig = plt.figure(figsize=(10, 7), num=str('SOFA Source Positions for ' + os.path.basename(in_sofa_file)))
        plot_coordinates(in_sofa_file, sofa_pos_fig)
        
        plotHRIR(in_sofa_file, legend, measurement, emitter)
    
    plotHRTF(in_sofa_file, legend, xlim, ylim, measurement, emitter)
    
    plt.show()
    
    plt.close()
    return

def saveSOFAGraphs(in_sofa_file, xlim:str, ylim:str, measurement:int=0, emitter:int=1):
    """
    Calls functions to plot source positions, HRIR, and HRTF for a given SOFA file, and saves them. Provides default values if they aren't given.

    Args:
        in_sofa_file (str): Path to sofa file to graph.
        xlim (str): Bounds for the x-axis, should be passed in the format [lower, upper] (e.g., [20, 20000]).
        ylim (str): Bounds for the y-axis, should be passed in the format [lower, upper] (e.g., [-150, 0]).
        measurement (int, optional): Measurement index to plot. Defaults to 0.
        emitter (int, optional): Emitter to plot. Defaults to 1.

    Returns:
    """    
    plt.close()
    
    if not xlim:
        xlim = '20, 20000'
    if not ylim:
        ylim = '-150, 0'
    if not measurement:
        measurement = 0
    if not emitter:
        emitter = 1
    legend = []
    
    export_directory = filedialog.askdirectory(title='Select Save Directory', initialdir=os.path.dirname(in_sofa_file[0]))
    if not export_directory:
        errorWindow(error_message='Directory not given.')
        return -1
    
    if sofa_mode_selection == 0:
        in_sofa_file = in_sofa_file[0]
        
        export_directory = os.path.join(export_directory, os.path.basename(in_sofa_file))
        export_directory = export_directory + '-measurements'
        try:
            os.mkdir(export_directory)
        except FileExistsError:
            pass
        
        # plot & save source coordinates
        sofa_pos_fig = plt.figure(figsize=(10, 7), num=str('SOFA Source Positions for ' + os.path.basename(in_sofa_file)))
        plot_coordinates(in_sofa_file, sofa_pos_fig)
        plt.savefig(os.path.join(export_directory, ('SOFA_Source_Positions_for_' + os.path.basename(in_sofa_file).replace(' ', '_') + '.png')))
        
        # plot & save HRIR
        plotHRIR(in_sofa_file, legend, measurement, emitter)
        plt.savefig(os.path.join(export_directory, ('Head-Related_Impulse_Response_at_M={0}_E={1}_for_'.format(measurement, emitter) + os.path.basename(in_sofa_file).replace(' ', '_') + '.png')))
    
    if sofa_mode_selection == 1:
        export_directory = os.path.join(export_directory, 'hrtf-comparison-measurements')
        try:
            os.mkdir(export_directory)
        except FileExistsError:
            pass
    
    # plot & save HRTF
    plotHRTF(in_sofa_file, legend, xlim, ylim, measurement, emitter)
    plt.savefig(os.path.join(export_directory, ('Left-Channel_HRTF_Comparison_at_M={0}_for_emitter_{1}'.format(measurement, emitter) + '.png')))
    
    plt.close()
    
    messageWindow('Successfully exported to:\n' + (os.path.basename(export_directory)), title='Export Successful', width=400, tooltip_text=export_directory)

def renderWithSOFA(angle:str, elev:str, in_source_file:str, in_sofa_file:str, target_fs:int=48000):
    """
    Renders a given source file with a given sofa file at the given azimuth and elevation. Targets a sampling rate of 48kHz.

    Args:
        angle (str): Desired azimuth to convolve the source file with.
        elev (str): Desired elevation to convolve the source file with.
        in_source_file (str): Path to source file.
        in_sofa_file (str): Path to sofa file.
        target_fs (int, optional): Target sampling rate. Defaults to 48000.

    Returns:
        np.ndarray: Convolved audio from azimuth and elevation.
    """    
    global Stereo3D
    try:
        isinstance(in_source_file, str)
    except NameError:
        errorWindow('NameError: Missing source file.')
        return
    except TypeError:
        errorWindow('TypeError: Missing source file.')
        return
    except not in_source_file:
        errorWindow('Empty Source File: Missing source file.')
        return
    except type(in_source_file) == None:
        errorWindow('NoneType: Missing source file.')
        return
    else:
        if not in_source_file:
            errorWindow('Missing source file.')
            return
        
        if not angle:
            angle = 0
        
        if not elev:
            elev = 0
        
        # init
        SOFA_HRTF = sofa.Database.open(in_sofa_file)
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
        [source_x, fs_x] = sf.read(in_source_file)
        # if it's not mono, make it mono.
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
        
        exportSOFAConvolved(in_source_file, in_sofa_file, angle_label, elev_label, Stereo3D, sofa_positions, int(target_fs))
        
        return Stereo3D

def exportSOFAConvolved(in_source_file:str, in_sofa_file:str, angle_label:int, elev_label:int, audioContent:np.ndarray, sofa_positions:np.ndarray, samplerate:int=48000):
    """
    Takes a given convolved audio, exports it to a .wav file, and displays relevant information to the user.

    Args:
        in_source_file (str): Path to source file.
        in_sofa_file (str): Path to sofa file.
        angle_label (int): Azimuth that was convolved.
        elev_label (int): Elevation that was convolved.
        audioContent (np.ndarray): Convolved audio in np.ndarray format.
        sofa_positions (np.ndarray): Position of the convolved point. Convention standard should be 1 meter, but will vary depending on measurement environments, 
        samplerate (int, optional): _description_. Defaults to 48000.

    Returns:
        _type_: _description_
    """    
    # find out where the select sofa file is, so it can be provided as a default directory for exporting instead of the home directory
    sofa_file_location = os.path.join(*in_sofa_file.split('/')[:-1])
    export_directory = filedialog.askdirectory(title='Select Save Directory', initialdir=sofa_file_location)
    
    # test for validity of SOFA file. technically, you could pass instead of return -1 here, but if the sofa file doesn't have correct distance information, then it may be malformed in other ways, too. 
    try:
        source_distance = sofa_positions[0,2]
    except:
        errorWindow(error_message='No distance information available.\n')
        return -1
    
    if export_directory:
        export_filename = str(os.path.join(str(export_directory), str(os.path.basename(in_sofa_file)[:-5]))) + '-' + str(os.path.basename(in_source_file)[:-4]) + '-azi_' + str(angle_label) + '-elev_' + str(elev_label) + '-export.wav'
        sf.write(export_filename, audioContent, samplerate=samplerate)
        messageWindow(message=('Using HRTF set: ' + str(os.path.basename(in_sofa_file)) + '\n' + 'Source distance is: ' + str(sofa_positions[0,2]) + 'meters\n\n' + 'Source: ' + str(os.path.basename(in_source_file)) + '\nrendered at azimuth ' + str(angle_label) + ' and elevation ' + str(elev_label)), title='SOFA Rendering', width=500, height=150, tooltip_text=str(export_filename))
    if not export_directory:
        errorWindow(error_message='\nNot rendered: Export directory not given.')
        return -1

def spectrogramWindow(audio_file_path:str):
    """
    Creates a window to configure the spectrogram with.

    Args:
        audio_file_path (str): Path to audio file to be passed to the spectrogram function.
    """    
    spectrogramConfigWindow = tk.Toplevel(root)
    spectrogramConfigWindow.iconphoto(False, icon_photo)
    centered_window(spectrogramConfigWindow)
    spectrogramConfigWindow.grid_columnconfigure(0, weight=1)
    spectrogramConfigWindow.grid_rowconfigure(0, weight=1)
    spectrogramConfigWindow.minsize(570, 250)
    spectrogramConfigWindow.title('Configure Spectrogram')
    spectrogramConfigWindowContentFrame = tk.Frame(spectrogramConfigWindow, borderwidth=10, relief='flat')
    spectrogramConfigWindowContentFrame.grid(row=0, column=0)

    spectrogramConfigTitleLabel = ttk.Label(spectrogramConfigWindowContentFrame, text='Configure Spectrogram', font=("TkDefaultFont", str(parse_font_dict['size'] + 2), "bold"))
    spectrogramConfigTitleLabel.grid(row=0, column=1)
    spectrogramAudioTitleLabel = ttk.Label(spectrogramConfigWindowContentFrame, text='Audio file:')
    spectrogramAudioTitleLabel.grid(row=1, column=0)
    spectrogramSelectedAudioLabel = ttk.Label(spectrogramConfigWindowContentFrame, text=shorten_file_name(os.path.basename(audio_file_path), 24))
    create_tooltip(spectrogramSelectedAudioLabel, text=str(audio_file_path))
    spectrogramSelectedAudioLabel.grid(row=1, column=2)
    startTimeAudioLabel = ttk.Label(spectrogramConfigWindowContentFrame, text='Start time (ms):')
    startTimeAudioLabel.grid(row=2, column=0)
    endTimeAudioLabel = ttk.Label(spectrogramConfigWindowContentFrame, text='End time (ms):')
    endTimeAudioLabel.grid(row=3, column=0)
    dynamicRangeMinLabel = ttk.Label(spectrogramConfigWindowContentFrame, text='Intensity scale min (dB):')
    dynamicRangeMinLabel.grid(row=4, column=0)
    dynamicRangeMaxLabel = ttk.Label(spectrogramConfigWindowContentFrame, text='Intensity scale max (dB):')
    dynamicRangeMaxLabel.grid(row=5, column=0)
    plotTitleLabel = ttk.Label(spectrogramConfigWindowContentFrame, text='Plot title:')
    plotTitleLabel.grid(row=6, column=0)

    startTimeStringVar = tk.StringVar()
    endTimeStringVar = tk.StringVar()
    dynamicRangeMinStringVar = tk.StringVar()
    dynamicRangeMaxStringVar = tk.StringVar()
    plotTitleStringVar = tk.StringVar()

    startTimeAudioEntry = ttk.Entry(spectrogramConfigWindowContentFrame, textvariable=startTimeStringVar)
    startTimeAudioEntry.grid(row=2, column=2)
    endTimeAudioEntry = ttk.Entry(spectrogramConfigWindowContentFrame, textvariable=endTimeStringVar)
    endTimeAudioEntry.grid(row=3, column=2)
    dynamicRangeMinEntry = ttk.Entry(spectrogramConfigWindowContentFrame, textvariable=dynamicRangeMinStringVar)
    dynamicRangeMinEntry.grid(row=4, column=2)
    dynamicRangeMaxEntry = ttk.Entry(spectrogramConfigWindowContentFrame, textvariable=dynamicRangeMaxStringVar)
    dynamicRangeMaxEntry.grid(row=5, column=2)
    plotTitleEntry = ttk.Entry(spectrogramConfigWindowContentFrame, textvariable=plotTitleStringVar)
    plotTitleEntry.grid(row=6, column=2)

    viewSpectrogramButton = ttk.Button(spectrogramConfigWindowContentFrame, text='View spectrogram', command=lambda:spectrogram(audio_file_path, startTimeStringVar.get(), endTimeStringVar.get(), dynamicRangeMinStringVar.get(), dynamicRangeMaxStringVar.get(), plotTitleStringVar.get()))
    viewSpectrogramButton.grid(row=7, column=1)

def spectrogram(audio_file_path:str, start_ms:str, end_ms:str, dynamic_range_min:str, dynamic_range_max:str, plot_title:str):
    """
    Plots spectrogram with given parameters audio_file_path, start_ms, end_ms, dynamic_range_min, dynamic_range_max, and plot_title. start_ms will default to 0ms, end_ms will default to the length of the given audio file in ms, dynamic_range_min and dynamic_range_max will default to auto-calculations by matplotlib, and plot_title defaults to the file name of the given audio file.

    Args:
        audio_file_path (str): Path to audio file to be plotted.
        start_ms (str): Start time of graph in milliseconds. Defaults to 0.
        end_ms (str): End time of graphs in milliseconds. Defaults to (len(samples)/samplerate) * 1000
        dynamic_range_min (str): Lower bound dynamic range of graph in dB. Defaults to auto-calculation by matplotlib.
        dynamic_range_max (str): Upper bound dynamic range of graph in dB. Defaults to auto-calculation by matplotlib.
        plot_title (str): Title of plot. Defaults to file name of given audio file.
    """    
    sr, samples = wavfile.read(str(audio_file_path))
    
    if not plot_title:
        plot_title = os.path.basename(audio_file_path)
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
    fig.canvas.manager.set_window_title(str('Spectrogram for ' + os.path.basename(audio_file_path)))
    
    plt.title(str(plot_title))
    plt.show()

def callback_url(url:str):
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
    
    nextButton = ttk.Button(tutorialWindowContentFrame, text='Next (SOFA Help) ->', command=lambda:sofaHelpPage())
    nextButton.grid(row=8, column=2, sticky='E')

def sofaHelpPage():
    clearWidgets(tutorialWindowContentFrame)
    sofaTitleTutorialLabel = tk.Label(tutorialWindowContentFrame, text='SOFA Functions\n', font=("TkDefaultFont", str(parse_font_dict['size'] + 4), "bold"))
    sofaTitleTutorialLabel.grid(row=0, column=1)
    
    selectSOFAFileTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Select SOFA File"\nPresents dialogue box for selecting .SOFA file(s).\n')
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
    
    viewSOFAFileTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"View SOFA File/View SOFA HRTF"\nTakes the above selected values\nand presents a 3D view of the .SOFA file,\n in addition to individual measurements\nfrom the .SOFA file.\nIf multiple .SOFA files are selected,\n the only graph displayed will be a layered HRTF graph.\n')
    viewSOFAFileTutorialLabel.grid(row=6, column=0)
    saveSOFAFileTutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Save all SOFA graphs/Save SOFA HRTF"\nSaves graphs/plots for source positions,\nhead-related impulse response, and head-related transfer function\nfor the provided azimuth, elevation, emitter, and measurement index to\nthe provided directory.\nIf multiple .SOFA files are selected,\n the only graph saved will be a layered HRTF graph.\n')
    saveSOFAFileTutorialLabel.grid(row=6, column=2)
    renderSOFATutorialLabel = tk.Label(tutorialWindowContentFrame, text='"Render Source with SOFA File"\nConvolves the source file with\nthe desired values in the .SOFA file.\nDisabled if multiple .SOFA files are selected.\n')
    renderSOFATutorialLabel.grid(row=7, column=1)
    
    prevButton = ttk.Button(tutorialWindowContentFrame, text='<- Previous (HRTF Help)', command=lambda:hrtfHelpPage())
    prevButton.grid(row=8, column=0, sticky='W')
    
    nextButton = ttk.Button(tutorialWindowContentFrame, text='Next (General Help) ->', command=lambda:generalHelpPage())
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
    * 'Source File Stereo -> Mono' must ALWAYS be pressed before convolving with HRTF!
    * You can use 'Tab' to select a subsequent text box.
    * The default values for SOFA functions are as follows:
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
    
    prevButton = ttk.Button(tutorialWindowContentFrame, text='<- Previous (SOFA Help)', command=lambda:sofaHelpPage())
    prevButton.grid(row=20, column=0, sticky='W')

root = tk.Tk()
root.minsize(565, 910)
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

titleLabel = ttk.Label(rootFrame, text="Garrett's Great\nHRTF Functions\n", justify='center', font=("TkDefaultFont", str(parse_font_dict['size'] + 2), "bold"))
titleLabel.grid(row=0, column=0, columnspan=3)

topSectionFrame = tk.Frame(rootFrame, borderwidth=10, relief='ridge')
topSectionFrame.grid(row=1, column=0, columnspan=3)

hrtfSourceSelectionFrame = tk.Frame(topSectionFrame, borderwidth=5, relief='flat')
hrtfSourceSelectionFrame.grid(row=0, column=0, columnspan=3)

hrtfFrame = tk.Frame(hrtfSourceSelectionFrame, borderwidth=10, relief='flat')
hrtfFrame.grid(row=0, column=0)

selectHRTFFileButton = ttk.Button(hrtfFrame, text='Select HRTF File (.wav)', command=lambda:selectHRTFFile())
selectHRTFFileButton.grid(row=0, column=0)
selectHRTFFileLabel = ttk.Label(hrtfFrame, text='HRTF file:\n', justify='center', wraplength=120)
selectHRTFFileLabel.grid(row=1, column=0)
getHRTFFileDataButton = ttk.Button(hrtfFrame, text='Get HRTF File Data', state='disabled', command=lambda:getHRTFFileData(hrtf_file, HRIR))
getHRTFFileDataButton.grid(row=3, column=0)
timeDomainVisualHRTFButton = ttk.Button(hrtfFrame, text='HRTF Time Domain Visualization', state='disabled', command=lambda:timeDomainVisualHRTF(hrtf_file, HRIR))
timeDomainVisualHRTFButton.grid(row=4, column=0)
freqDomainVisualHRTFButton = ttk.Button(hrtfFrame, text='HRTF Frequency Domain Visualization', state='disabled', command=lambda:freqDomainVisualHRTF(hrtf_file))
freqDomainVisualHRTFButton.grid(row=5, column=0)

sourceFrame = tk.Frame(hrtfSourceSelectionFrame, borderwidth=10, relief='flat')
sourceFrame.grid(row=0, column=2)
selectSourceFileButton = ttk.Button(sourceFrame, text='Select source file (.wav)', command=lambda:selectSourceFile())
selectSourceFileLabel = ttk.Label(sourceFrame, text='Source file:\n', justify='center', wraplength=120)
selectSourceFileButton.grid(row=0, column=2)
selectSourceFileLabel.grid(row=1, column=2)
getSourceFileDataButton = ttk.Button(sourceFrame, text='Get Source File Data', state='disabled', command=lambda:getSourceFileData(source_file))
getSourceFileDataButton.grid(row=2, column=2)
spectrogramButton = ttk.Button(sourceFrame, text='View Spectrogram', state='disabled', command=lambda:spectrogramWindow(source_file))
spectrogramButton.grid(row=3, column=2)
stereoToMonoButton = ttk.Button(sourceFrame, text='Source File Stereo -> Mono', state='disabled', command=lambda:stereoToMono(sig))
stereoToMonoButton.grid(row=4, column=2)

hrtfOperationsFrame = tk.Frame(topSectionFrame, borderwidth=10, relief='flat')
hrtfOperationsFrame.grid(row=1, column=1)

resampleButton = ttk.Button(hrtfOperationsFrame, text='Resample', state='disabled', command=lambda:fs_resample(sig_mono, fs_s, HRIR, fs_H))
resampleButton.grid(row=1, column=1)
timeDomainConvolveButton = ttk.Button(hrtfOperationsFrame, text='Time Domain Convolve', state='disabled', command=lambda:timeDomainConvolve(sig_mono, HRIR))
timeDomainConvolveButton.grid(row=2, column=1)
exportConvolvedButton = ttk.Button(hrtfOperationsFrame, text='Export Convolved', state='disabled', command=lambda:exportConvolved(Bin_Mix, fs_s, source_file, hrtf_file))
exportConvolvedButton.grid(row=3, column=1)

sectionalLabel = ttk.Label(rootFrame, text='\n')
sectionalLabel.grid(row=2, column=0, columnspan=3)

sofaLabel = ttk.Label(rootFrame, text="Garrett's Great\nSOFA Functions\n", justify='center', font=("TkDefaultFont", str(parse_font_dict['size'] + 2), "bold"))
sofaLabel.grid(row=3, column=0, columnspan=3)

bottomSectionFrame = tk.Frame(rootFrame, borderwidth=10, relief='ridge')
bottomSectionFrame.grid(row=4, column=0, columnspan=3)

selectSOFAFileButton = ttk.Button(bottomSectionFrame, text='Select SOFA File', command=lambda:selectSOFAFile())
selectSOFAFileButton.grid(row=0, column=0, columnspan=3)
selectSOFAFileLabel = ttk.Label(bottomSectionFrame, text='SOFA file:\n', justify='center', wraplength=240)
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

getSOFAFileMetadataButton = ttk.Button(bottomLeftFrame, text='Get SOFA File Metadata', state='disabled', command=lambda:getSOFAFileMetadata(sofa_file_path_list[0]))
getSOFAFileMetadataButton.grid(row=0, column=0)
getSOFAFileDimensionsButton = ttk.Button(bottomRightFrame, text='Get SOFA File Dimensions', state='disabled', command=lambda:getSOFAFileDimensions(sofa_file_path_list[0]))
getSOFAFileDimensionsButton.grid(row=0, column=0)

sofaMeasurementTextBox = ttk.Entry(bottomLeftFrame, state='disabled', width=5, textvariable=sofaMeasurementStringVar)
sofaMeasurementTextBox.grid(row=1, column=0)
sofaMeasurementLabel = ttk.Label(bottomLeftFrame, text='Measurement Index\n(default: 0)\n', justify='center')
sofaMeasurementLabel.grid(row=2, column=0)
sofaEmitterTextBox = ttk.Entry(bottomRightFrame, state='disabled', width=5, textvariable=sofaEmitterStringVar)
sofaEmitterTextBox.grid(row=1, column=0)
sofaEmitterLabel = ttk.Label(bottomRightFrame, text='Emitter\n(default: 1)\n', justify='center')
sofaEmitterLabel.grid(row=2, column=0)

frequencyXLimTextBox = ttk.Entry(bottomLeftFrame, state='disabled', width=15, textvariable=freqXLimStringVar)
frequencyXLimTextBox.grid(row=3, column=0)
frequencyXLimLabel = ttk.Label(bottomLeftFrame, text='Frequency Range (Hz)\n[start, end]', justify='center')
frequencyXLimLabel.grid(row=4, column=0)
magnitudeYLimTextBox = ttk.Entry(bottomRightFrame, state='disabled', width=15, textvariable=magYLimStringVar)
magnitudeYLimTextBox.grid(row=3, column=0)
magnitudeYLimLabel = ttk.Label(bottomRightFrame, text='Magnitude (dB)\n[start, end]', justify='center')
magnitudeYLimLabel.grid(row=4, column=0)

azimuthTextBox = ttk.Entry(bottomLeftFrame, state='disabled', width=5, textvariable=azimuthStringVar)
azimuthTextBox.grid(row=5, column=0)
azimuthLabel = ttk.Label(bottomLeftFrame, text='Desired azimuth (in deg)', justify='center')
azimuthLabel.grid(row=6, column=0)
elevationTextBox = ttk.Entry(bottomRightFrame, state='disabled', width=5, textvariable=elevationStringVar)
elevationTextBox.grid(row=5, column=0)
elevationLabel = ttk.Label(bottomRightFrame, text='Desired elevation (in deg)', justify='center')
elevationLabel.grid(row=6, column=0)

sofaViewButton = ttk.Button(bottomSectionFrame, text='View SOFA File', state='disabled', command=lambda:viewSOFAGraphs(sofa_file_path_list, freqXLimStringVar.get(), magYLimStringVar.get(), sofaMeasurementStringVar.get(), sofaEmitterStringVar.get()))
sofaViewButton.grid(row=4, column=0, columnspan=2)
sofaSaveButton = ttk.Button(bottomSectionFrame, text='Save all SOFA graphs', state='disabled', command=lambda:saveSOFAGraphs(sofa_file_path_list, freqXLimStringVar.get(), magYLimStringVar.get(), sofaMeasurementStringVar.get(), sofaEmitterStringVar.get()))
sofaSaveButton.grid(row=4, column=1, columnspan=2)
sofaRenderButton = ttk.Button(bottomSectionFrame, text='Render Source with SOFA file', state='disabled', command=lambda:renderWithSOFA(azimuthStringVar.get(), elevationStringVar.get(), source_file, sofa_file_path_list[0]))
sofaRenderButton.grid(row=5, column=0, columnspan=3)

tutorialButton = ttk.Button(rootFrame, text='Help', command=lambda:createHelpWindow())
tutorialButton.grid(row=6, column=0, sticky='W')

quitButton = ttk.Button(rootFrame, text='Quit', command=lambda:quit())
quitButton.grid(row=6, column=2, sticky='E')

# prevents the window from appearing at the bottom of the stack
root.focus_force()

root.protocol('WM_DELETE_WINDOW', lambda:quit())

sv_ttk.set_theme(darkdetect.theme())

root.mainloop()