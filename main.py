# much of this is sourced from andrea genovese
# please find him here: https://andreagenovese.com/
# and his wonderful tutorial about hrtf processing here: https://www.youtube.com/watch?v=a4mpK_2koR4

# also please keep in mind that this isn't supposed to be "efficient" or "clean" or "lightweight".
# this is meant to be the most brute force way to do all my hrtf processing in one file with one interface.

import sys  # <- replacement for pythonic quit(), which doesn't play nicely with cx_freeze
import os  # <- reading files from disk, adapting to differing os directory path conventions
import tempfile  # <- adapting to differing os temp file locations
from unittest.mock import MagicMock # <- trying to get librosa to run without numba so i can use nuitka for compile; see https://github.com/librosa/librosa/issues/1854#event-18920426125

# mock numba module
sys.modules['numba'] = MagicMock()

# no-op decorator w arguments
def no_op_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0] # jit w/o args
    else:
        def wrapper(func):
            return func # jit w/ args
        return wrapper

sys.modules['numba'].jit == no_op_decorator

os.environ["LIBROSA_CACHE_DIR"] = str(
    tempfile.gettempdir()
)  # <- must be called before importing librosa
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = (
    "hide"  # <- gets rid of pygame welcome message, which clutters up cli
)

import numpy as np  # <- matrix calc & more (but mostly matrix calc)
import matplotlib.pyplot as plt  # <- data visualization
import soundfile as sf  # <- read audio into ndarray
import sofa  # <- read SOFA HRTFs
import librosa  # <- resample function
from scipy import signal  # <- fast convolution function
from scipy.io import wavfile  # <- used for spectrogram
import pygame  # <- for playing audio files directly
import tkinter as tk  # <- reliable, if clunky, gui
from tkinter import (
    ttk,
)  # <- necessary so buttons don't turn invisible with dark mode enabled
from tkinter import filedialog  # <- gui file selection from disk
from tkinter import font  # <- ensures fonts are system-compatible
import webbrowser  # <- help page links
import sv_ttk  # <- handles ttk
import darkdetect  # <- detects os light/dark mode
from base64 import b64decode  # <- decode bitmap image backup for asset

if sys.platform == "win32":
    import pywinstyles # <- aesthetics on dark mode for windows only
    try:
        import matplotlib
        import pyi_splash # <- windows-only splash screen (needed for pyinstaller, bc pyinstaller wipes the matplotlib cache everytime the application closes, and cx_freeze doesn't build to windows with python 3.13)
        pyi_splash.close()
        matplotlib.use('TkAgg')
    except:
        pass

if sys.platform == "darwin":
    pass

if sys.platform == "linux":
    pass

librosa.cache.clear(warn=False)

source_file = None

class ToolTip(
    object
):  # https://stackoverflow.com/questions/20399243/display-message-when-hovering-over-something-with-mouse-cursor-in-python
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def show_tooltip(self, text):
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        tooltip_label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            foreground="black",
            relief="solid",
            borderwidth=0,
            font="TkDefaultFont 10 normal",
        )
        tooltip_label.pack(ipadx=1)

    def hide_tooltip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def centered_window(
    window,
):  # https://www.geeksforgeeks.org/how-to-center-a-window-on-the-screen-in-tkinter/
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

    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)


def errorWindow(
    error_message: str = "Generic Error Message",
    title: str = "Error",
    width: int = 300,
    height: int = 120,
    **kwargs,
):
    """
    Creates an error window.

    Args:
        error_message (str, optional): Error message to be displayed. Defaults to 'Generic Error Message'.
        title (str, optional): Title of window. Defaults to 'Error'.
        width (int, optional): Width of window. Defaults to 300.
        height (int, optional): Height of window. Defaults to 120.

    Keyword Arguments:
        tooltip_text (str, optional): If a string is provided, a tooltip will appear with that string when hovering over the error message.

    Returns:

    """
    tooltip_text = kwargs.get("tooltip_text", None)
    errorWindow = tk.Toplevel(root)
    errorWindow.iconphoto(False, icon_photo)
    centered_window(errorWindow)
    errorWindow.title(str(title))
    errorWindow.geometry(str(width) + "x" + str(height))
    errorWindow.minsize(width, height)
    errorMessageLabel = ttk.Label(
        errorWindow, text="\nError: " + str(error_message) + "\n", justify="center"
    )
    if tooltip_text:
        create_tooltip(errorMessageLabel, text=str(tooltip_text))
    errorMessageLabel.pack()
    errorConfirmButton = ttk.Button(
        errorWindow,
        text="OK",
        style="my.TButton",
        command=lambda: errorWindow.destroy(),
    )
    errorConfirmButton.pack()
    errorWindow.focus_force()
    apply_theme_to_titlebar(errorWindow)
    return -1

def messageWindow(
    message: str = "Generic Message",
    title: str = "Title",
    width: int = 300,
    height: int = 120,
    **kwargs,
):
    """
    Creates a message/alert window.

    Args:
        message (str, optional): Message to be displayed. Defaults to 'Generic Message'.
        title (str, optional): Title of window. Defaults to 'Title'.
        width (int, optional): Width of window. Defaults to 300.
        height (int, optional): Height of window. Defaults to 120.

    Keyword Arguments:
        tooltip_text (str, optional): If a string is provided, a tooltip will appear with that string when hovering over the message.

    Returns:

    """
    tooltip_text = kwargs.get("tooltip_text", None)
    messageWindow = tk.Toplevel(root)
    messageWindow.iconphoto(False, icon_photo)
    centered_window(messageWindow)
    messageWindow.title(str(title))
    messageWindow.geometry(str(width) + "x" + str(height))
    messageWindow.minsize(width, height)
    messageLabel = ttk.Label(
        messageWindow, text="\n" + str(message) + "\n", justify="center"
    )
    if tooltip_text:
        create_tooltip(messageLabel, text=str(tooltip_text))
    messageLabel.pack()
    messageConfirmButton = ttk.Button(
        messageWindow,
        text="OK",
        style="my.TButton",
        command=lambda: messageWindow.destroy(),
    )
    messageConfirmButton.pack()
    messageWindow.focus_force()
    apply_theme_to_titlebar(messageWindow)
    return -1


def shorten_file_name(old_filename: str, num_shown_char: int):
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
        new_filename = str(old_filename[: int(num_shown_char)]) + "..."
    else:
        new_filename = old_filename
    return new_filename


def playAudio(path_to_audio_file: str):
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
    hrtf_file = filedialog.askopenfilename(filetypes=[(".wav files", ".wav")])
    root.focus_force()
    try:
        if hrtf_file:
            [HRIR, fs_H] = sf.read(hrtf_file)
    except RuntimeError:
        errorWindow(
            "\nError loading file:\n\n"
            + str(os.path.basename(hrtf_file))
            + "\n\nRuntime error.\nDoes this file exist on the local drive?",
            title="Error",
            width=300,
            height=220,
            tooltip_text=hrtf_file,
        )
        return
    else:
        if hrtf_file:
            hrtf_file_print = hrtf_file.split("/")
            selectHRTFFileLabel.config(
                text="HRTF file:\n" + shorten_file_name(os.path.basename(hrtf_file), 13)
            )
            create_tooltip(selectHRTFFileLabel, text=str(hrtf_file))
            getHRTFFileDataButton.config(state="active")
            timeDomainVisualHRTFButton.config(state="active")
            freqDomainVisualHRTFButton.config(state="active")
            resampleButton.config(state="disabled")
            # root_menu_file_hrtf.entryconfigure(1, state='normal') # separator
            root_menu_file_hrtf.entryconfigure(2, state='normal')
            root_menu_file_hrtf.entryconfigure(3, state='normal')
            root_menu_file_hrtf.entryconfigure(4, state='normal')
            [HRIR, fs_H] = sf.read(hrtf_file)
        else:
            return


def selectSourceFile():
    global source_file
    global source_file_print
    global sig
    global fs_s
    source_file = filedialog.askopenfilename(filetypes=[(".wav files", ".wav")])
    root.focus_force()
    # check to see if the selected file is on the drive (in the case of network drives, for example)
    try:
        if source_file:
            [sig, fs_s] = sf.read(source_file)
    except RuntimeError:
        errorWindow(
            "\nError loading file:\n\n"
            + str(os.path.basename(source_file))
            + "\n\nRuntime error.\nDoes this file exist on the local drive?",
            title="Error",
            width=300,
            height=220,
            tooltip_text=source_file,
        )
        return
    else:
        if source_file == "":
            return
        source_file_print = source_file.split("/")
        selectSourceFileLabel.config(
            text="Source file:\n" + shorten_file_name(os.path.basename(source_file), 13)
        )
        create_tooltip(selectSourceFileLabel, text=str(source_file))
        getSourceFileDataButton.config(state="active")
        stereoToMonoButton.config(state="active")
        spectrogramButton.config(state="active")
        resampleButton.config(state="disabled")
        # root_menu_file_source.entryconfig(1, state='normal') # separator
        root_menu_file_source.entryconfig(2, state='normal')
        root_menu_file_source.entryconfig(3, state='normal')
        root_menu_file_source.entryconfig(4, state='normal')
        [sig, fs_s] = sf.read(source_file)


def getHRTFFileData(in_hrtf_file: str, in_HRIR: np.ndarray):
    """
    For a given HRTF file, create a window, read and display the sample rate and data dimensions, and create buttons to play & pause the HRTF, and close the window.

    Args:
        in_hrtf_file (str): Path to HRTF file (.wav).
        in_HRIR (np.ndarray): HRIR object pulled from in_hrtf_file.
    """
    if len(in_HRIR.shape) < 2:
        errorHrtfFileData = [
            str("Selected file: " + os.path.basename(in_hrtf_file)),
            str("Sample rate: " + str(fs_H)),
            str("Data dimensions: " + str(in_HRIR.shape)),
        ]
        errorWindow(
            error_message="Selected file only has one dimension (mono channel).\nAre you sure this is an HRTF/HRIR?\nHover for more info.",
            width=400,
            height=140,
            tooltip_text="\n".join(map(str, errorHrtfFileData)),
        )
        return -1
    else:
        hrtfFileDataWindow = tk.Toplevel(root)
        apply_theme_to_titlebar(hrtfFileDataWindow)
        hrtfFileDataWindow.iconphoto(False, icon_photo)
        centered_window(hrtfFileDataWindow)
        hrtfFileDataWindow.minsize(300, 210)
        hrtfFileDataWindow.title("HRTF File Data")
        windowTitleHRTFData = ttk.Label(
            hrtfFileDataWindow,
            text="\n" + os.path.basename(in_hrtf_file) + "\n",
            justify="center",
        )
        windowTitleHRTFData.pack()
        create_tooltip(windowTitleHRTFData, in_hrtf_file)
        hrtfSampleRateLabel = ttk.Label(
            hrtfFileDataWindow, text="Sample rate: " + str(fs_H), justify="center"
        )
        hrtfSampleRateLabel.pack()
        hrtfDataDimLabel = ttk.Label(
            hrtfFileDataWindow,
            text="Data dimensions: " + str(in_HRIR.shape) + "\n",
            justify="center",
        )
        hrtfDataDimLabel.pack()
        hrtfPlayFileButton = ttk.Button(
            hrtfFileDataWindow,
            text="Play HRTF",
            style="my.TButton",
            command=lambda: playAudio(in_hrtf_file),
        )
        hrtfPlayFileButton.pack()
        hrtfPauseFileButton = ttk.Button(
            hrtfFileDataWindow,
            text="Pause HRTF",
            style="my.TButton",
            command=lambda: pygame.mixer.music.pause(),
        )
        hrtfPauseFileButton.pack()
        hrtfFileDataCloseWindowButton = ttk.Button(
            hrtfFileDataWindow,
            text="Close",
            style="my.TButton",
            command=lambda: stopAudioAndCloseWindow(hrtfFileDataWindow),
        )
        hrtfFileDataCloseWindowButton.pack()


def getSourceFileData(in_source_file: str):
    """
    For a given audio file, create a window, read and display the sample rate and data dimensions, and create buttons to play & pause the audio, and close the window.

    Args:
        in_source_file (str): Path to audio file (.wav).
    """
    sourceFileDataWindow = tk.Toplevel(root)
    apply_theme_to_titlebar(sourceFileDataWindow)
    sourceFileDataWindow.iconphoto(False, icon_photo)
    centered_window(sourceFileDataWindow)
    sourceFileDataWindow.minsize(300, 210)
    sourceFileDataWindow.title("Source File Data")
    windowTitleSourceData = ttk.Label(
        sourceFileDataWindow,
        text="\n" + os.path.basename(in_source_file) + "\n",
        justify="center",
    )
    windowTitleSourceData.pack()
    create_tooltip(windowTitleSourceData, in_source_file)
    sourceSampleRateLabel = ttk.Label(
        sourceFileDataWindow, text="Sample rate: " + str(fs_s), justify="center"
    )
    sourceSampleRateLabel.pack()
    sourceDataDimLabel = ttk.Label(
        sourceFileDataWindow,
        text="Data dimensions: " + str(sig.shape) + "\n",
        justify="center",
    )
    sourceDataDimLabel.pack()
    sourcePlayFileButton = ttk.Button(
        sourceFileDataWindow,
        text="Play Source File",
        style="my.TButton",
        command=lambda: playAudio(in_source_file),
    )
    sourcePlayFileButton.pack()
    sourcePauseFileButton = ttk.Button(
        sourceFileDataWindow,
        text="Pause Source File",
        style="my.TButton",
        command=lambda: pygame.mixer.music.pause(),
    )
    sourcePauseFileButton.pack()
    sourceFileDataCloseWindowButton = ttk.Button(
        sourceFileDataWindow,
        text="Close",
        style="my.TButton",
        command=lambda: (stopAudioAndCloseWindow(sourceFileDataWindow)),
    )
    sourceFileDataCloseWindowButton.pack()


def stopAudioAndCloseWindow(window_to_close):
    # cannot believe i'm actually making this function
    try:
        pygame.mixer.music.pause()
    except pygame.error:
        pass
    finally:
        window_to_close.destroy()


def timeDomainVisualHRTF(in_hrtf_file: str, in_HRIR: np.ndarray):
    """
    For a given HRTF, plot the time domain visualization.

    Args:
        in_hrtf_file (str): Path to HRTF file (.wav).
        in_HRIR (np.ndarray): HRIR object pulled from in_hrtf_file.
    """
    plt.figure(num=str("Time Domain for " + os.path.basename(in_hrtf_file)))
    plt.plot(in_HRIR[:, 0])  # left channel data
    plt.plot(in_HRIR[:, 1])  # right channel data
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("HRIR at angle: " + os.path.basename(in_hrtf_file))
    plt.legend(["Left", "Right"])
    plt.show()


def freqDomainVisualHRTF(in_hrtf_file: str):
    """
    For a given HRTF, plot the frequency domain visualization.

    Args:
        in_hrtf_file (str): Path to HRTF file (.wav).
    """
    nfft = len(HRIR) * 8
    HRTF = np.fft.fft(HRIR, n=nfft, axis=0)
    HRTF_mag = (2 / nfft) * np.abs(HRTF[0 : int(len(HRTF) / 2) + 1, :])
    HRTF_mag_dB = 20 * np.log10(HRTF_mag)

    f_axis = np.linspace(0, fs_H / 2, len(HRTF_mag_dB))
    plt.figure(num=str("Frequency Domain for " + os.path.basename(in_hrtf_file)))
    plt.semilogx(f_axis, HRTF_mag_dB)
    plt.grid()
    plt.grid(which="minor", color="0.9")
    plt.title("HRTF at angle: " + os.path.basename(in_hrtf_file))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend(["Left", "Right"])
    plt.show()


def stereoToMono(in_sig: np.ndarray):
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
        messageWindow(
            message=("New source data dimensions:\n" + str(sig_mono.shape)),
            title="Stereo -> Mono",
            width=350,
            height=130,
        )
    else:
        sig_mono = in_sig
        messageWindow(
            message="Source file is already mono.", title="Stereo -> Mono", width=350
        )
    
    if getHRTFFileDataButton["state"] == tk.ACTIVE or getHRTFFileDataButton["state"] == 'active' or str(getHRTFFileDataButton["state"]) == 'active':
        resampleButton.config(state="active")



def fs_resample(s1: np.ndarray, f1: int, s2: np.ndarray, f2: int):
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

    messageWindow(
        message=(
            "Resampled at: "
            + str(fmax)
            + "Hz\n\n"
            + "Signal/source dimensions: "
            + str(s1.shape)
            + "\n"
            + "HRIR Dimensions: "
            + str(s2.shape)
        ),
        title="Resample",
        width=250,
        height=170,
    )

    timeDomainConvolveButton.config(state="active")
    return s1, f1, s2, f2


def timeDomainConvolve(in_sig_mono: np.ndarray, in_HRIR: np.ndarray):
    """
    For a given mono signal, time domain convolve it by the HRIR.

    Args:
        in_sig_mono (np.ndarray): Mono signal in np.ndarray format.
        in_HRIR (np.ndarray): HRIR signal in np.ndarray format.
    """
    global Bin_Mix
    s_L = np.convolve(in_sig_mono, in_HRIR[:, 0])
    s_R = np.convolve(in_sig_mono, in_HRIR[:, 1])

    Bin_Mix = np.vstack([s_L, s_R]).transpose()

    messageWindow(
        message=("New data dimensions: " + str(Bin_Mix.shape)),
        title="Time Domain Convolve",
        width=250,
    )

    exportConvolvedButton.config(state="active")


# freq domain convolution goes here at some point


def exportConvolved(
    in_Bin_Mix: np.ndarray, in_fs_s: int, in_source_file: str, in_hrtf_file: str
):
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
    export_directory = filedialog.askdirectory(
        title="Select Save Directory", initialdir=source_file_location
    )
    convolved_filename = (
        str(os.path.basename(in_source_file)[:-4])
        + "-"
        + str(os.path.basename(in_hrtf_file)[:-4])
        + "-export.wav"
    )
    if export_directory:
        export_filename = os.path.join(str(export_directory), str(convolved_filename))
        sf.write(export_filename, in_Bin_Mix, in_fs_s)
    if not export_directory:
        return -1

    if os.path.exists(export_filename):
        messageWindow(
            message=("File successfully exported as:\n" + convolved_filename),
            title="Export Successful",
            width=400,
            tooltip_text=str(export_filename),
        )
    else:
        errorWindow(error_message="Export Failed")


def selectSOFAFile():
    global sofa_file_print
    global sofa_file_path_list
    global sofa_mode_selection

    sofa_file_path_list = filedialog.askopenfilenames(
        filetypes=[(".SOFA files", ".sofa")]
    )
    root.focus_force()
    if len(sofa_file_path_list) > 1:
        sofa_mode_selection = 1
        for file in sofa_file_path_list:
            if file:
                try:
                    metadata_test = sofa.Database.open(file).Metadata.list_attributes()
                except OSError:
                    errorWindow(
                        "\nError loading file:\n\n"
                        + str(os.path.basename(file))
                        + "\n\nOS error.\nDoes this file exist on the local drive?\n\nAlternatively, does this SOFA file\ncontain correct metadata?",
                        title="Error",
                        width=300,
                        height=270,
                        tooltip_text=file,
                    )
                    return
                else:
                    continue
        selectSOFAFileLabel.config(text="SOFA files:\nHover to see selected files.")
        create_tooltip(
            selectSOFAFileLabel,
            (
                "NOTE: Graph viewing only - Convolution disabled when multiple SOFA files are selected.\n"
                + "\n".join(map(str, sofa_file_path_list))
            ),
        )
        getSOFAFileMetadataButton.config(state="disabled")
        getSOFAFileDimensionsButton.config(state="disabled")
        azimuthTextBox.config(state="disabled")
        elevationTextBox.config(state="disabled")
        sofaRenderButton.config(state="disabled")
        sofaViewButton.config(text="View SOFA HRTF")
        root_menu_file_sofa.entryconfig(5, label='View SOFA HRTF', state='normal')
        sofaSaveButton.config(text="Save SOFA HRTF...")
        root_menu_file_sofa.entryconfig(6, label='Save SOFA HRTF...', state='normal')

    if len(sofa_file_path_list) == 1:
        sofa_mode_selection = 0
        try:
            if sofa_file_path_list[0]:
                metadata_test = sofa.Database.open(
                    sofa_file_path_list[0]
                ).Metadata.list_attributes()
        except OSError:
            errorWindow(
                "\nError loading file:\n\n"
                + str(os.path.basename(sofa_file_path_list[0]))
                + "\n\nOS error.\nDoes this file exist on the local drive?\n\nAlternatively, does this SOFA file\ncontain correct metadata?",
                title="Error",
                width=300,
                height=270,
                tooltip_text=sofa_file_path_list[0],
            )
            return
        else:
            if sofa_file_path_list[0]:
                sofa_file_print = sofa_file_path_list[0].split("/")
                selectSOFAFileLabel.config(
                    text="SOFA file:\n"
                    + shorten_file_name(os.path.basename(sofa_file_path_list[0]), 20)
                )
                create_tooltip(selectSOFAFileLabel, text=str(sofa_file_path_list[0]))
                getSOFAFileMetadataButton.config(state="active")
                getSOFAFileDimensionsButton.config(state="active")
                azimuthTextBox.config(state="normal")
                elevationTextBox.config(state="normal")
                sofaRenderButton.config(state="active")
                sofaViewButton.config(text="View SOFA File")
                sofaSaveButton.config(text="Save all SOFA graphs")
                # root_menu_file_sofa.entryconfig(1, state='normal') # separator
                root_menu_file_sofa.entryconfig(2, state='normal')
                root_menu_file_sofa.entryconfig(3, state='normal')
                # root_menu_file_sofa.entryconfig(4, state='normal') # separator
                root_menu_file_sofa.entryconfig(5, state='normal')
                root_menu_file_sofa.entryconfig(6, state='normal')
                # root_menu_file_sofa.entryconfig(7, state='normal') # separator
                root_menu_file_sofa.entryconfig(8, state='normal')
            else:
                return

    if sofa_file_path_list:
        sofaViewButton.config(state="active")
        sofaSaveButton.config(state="active")
        sofaMeasurementTextBox.config(state="normal")
        sofaEmitterTextBox.config(state="normal")
        frequencyXLimTextBox.config(state="normal")
        magnitudeYLimTextBox.config(state="normal")


def getSOFAFileMetadata(in_sofa_file: str):
    """
    Retrieve, parse, and display in a new window the metadata of a given SOFA file.

    Args:
        in_sofa_file (str): Path to SOFA file.
    """
    sofaMetadataWindow = tk.Toplevel(root)
    apply_theme_to_titlebar(sofaMetadataWindow)
    sofaMetadataWindow.iconphoto(False, icon_photo)
    centered_window(sofaMetadataWindow)
    sofaMetadataWindow.geometry("400x400")
    sofaMetadataWindow.title("SOFA File Metadata")
    v = tk.Scrollbar(sofaMetadataWindow, orient="vertical")
    v.pack(side="right", fill="y")
    myString = ""
    for attr in sofa.Database.open(in_sofa_file).Metadata.list_attributes():
        myString = (
            myString
            + (
                "{0}: {1}".format(
                    attr, sofa.Database.open(in_sofa_file).Metadata.get_attribute(attr)
                )
            )
            + "\n"
        )
    windowSOFAMetadataLabel = tk.Text(
        sofaMetadataWindow, width=100, height=100, wrap="word", yscrollcommand=v.set
    )
    windowSOFAMetadataLabel.insert("end", str(myString))
    windowSOFAMetadataLabel.pack()

    v.config(command=windowSOFAMetadataLabel.yview)


def getSOFAFileDimensions(in_sofa_file: str):
    """
    Retrieve, parse, and display in a new window the dimensions of a given SOFA file.

    Args:
        in_sofa_file (str): Path to SOFA file.
    """
    sofaDimensionsWindow = tk.Toplevel(root)
    apply_theme_to_titlebar(sofaDimensionsWindow)
    sofaDimensionsWindow.iconphoto(False, icon_photo)
    centered_window(sofaDimensionsWindow)
    sofaDimensionsWindow.geometry("600x400")
    sofaDimensionsWindow.title("SOFA File Dimensions")
    v = tk.Scrollbar(sofaDimensionsWindow, orient="vertical")
    v.pack(side="right", fill="y")
    definitionsLabel = tk.Label(
        sofaDimensionsWindow,
        text="C = Size of coordinate dimension (always three).\n\nI = Single dimension (always one).\n\nM = Number of measurements.\n\nR = Number of receivers or SH coefficients (depending on ReceiverPosition_Type).\n\nE = Number of emitters or SH coefficients (depending on EmitterPosition_Type).\n\nN = Number of samples, frequencies, SOS coefficients (depending on self.GLOBAL_DataType).",
    )
    definitionsLabel.pack()
    myString = ""
    for dimen in sofa.Database.open(in_sofa_file).Dimensions.list_dimensions():
        myString = (
            myString
            + (
                "{0}: {1}".format(
                    dimen,
                    sofa.Database.open(in_sofa_file).Dimensions.get_dimension(dimen),
                )
            )
            + "\n"
        )
    windowSOFADimensionsLabel = tk.Text(
        sofaDimensionsWindow, width=10, height=10, wrap="word", yscrollcommand=v.set
    )
    windowSOFADimensionsLabel.insert("end", str(myString))
    windowSOFADimensionsLabel.pack()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def sanitizeBounds(bounds: str):
    """
    Sanitizes and splits bounds from the user-input frequency and magnitude ranges to account for whitespace, brackets, and commas.

    Args:
        bounds (str): Lower and upper bound to be sanitized and split.

    Returns:
        str: lower_limit, upper_limit
    """
    bounds = str(bounds)
    for char in bounds:
        if char == "[" or char == "]" or char == " ":
            bounds = bounds.replace(char, "")
    for i in range(0, len(bounds)):
        if bounds[i] == ",":
            lower_limit = bounds[0:i]
            upper_limit = bounds[i + 1 :]

    return lower_limit, upper_limit


def plot_coordinates(in_sofa_file: str, fig: plt.Figure):
    """
    Plots source coordinate positions of a SOFA file.

    Args:
        in_sofa_file (str): Path to SOFA file to be plotted.
        fig (matplotlib.figure.Figure): Figure for the data to be plotted on.

    Returns:
        mpl_toolkits.mplot3d.art3d.Line3DCollection: 3D plot data
    """
    SOFA_HRTF = sofa.Database.open(in_sofa_file)
    x0 = SOFA_HRTF.Source.Position.get_values(system="cartesian")
    n0 = x0
    ax = fig.add_subplot(111, projection="3d")
    q = ax.quiver(
        x0[:, 0], x0[:, 1], x0[:, 2], n0[:, 0], n0[:, 1], n0[:, 2], length=0.1
    )
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Source positions for: " + str(os.path.basename(in_sofa_file)))
    return q


def computeHRIR(in_sofa_file: str, measurement: int):
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
    t = np.arange(0, SOFA_HRTF.Dimensions.N) * SOFA_HRTF.Data.SamplingRate.get_values(
        indices={"M": measurement}
    )

    return t, receiver_dimensions, SOFA_HRTF.Data.IR


def computeHRTF(in_sofa_file: str, measurement: int, emitter: int):
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

    nfft = (
        len(
            SOFA_HRTF.Data.IR.get_values(
                indices={"M": measurement, "R": receiver, "E": emitter}
            )
        )
        * 8
    )
    HRTF = np.fft.fft(
        SOFA_HRTF.Data.IR.get_values(
            indices={"M": measurement, "R": receiver, "E": emitter}
        ),
        n=nfft,
        axis=0,
    )
    HRTF_mag = (2 / nfft) * np.abs(HRTF[0 : int(len(HRTF) / 2) + 1])
    HRTF_mag_dB = 20 * np.log10(HRTF_mag)
    f_axis = np.linspace(
        0,
        (
            SOFA_HRTF.Data.SamplingRate.get_values(
                indices={"M": measurement, "R": receiver, "E": emitter}
            )
        )
        / 2,
        len(HRTF_mag_dB),
    )

    return f_axis, HRTF_mag_dB, receiver_legend


def plotHRIR(in_sofa_file, legend: list, measurement: int, emitter: int):
    """
    Plots a head-related impulse response graph for a given .sofa file with a given legend, measurement index, and emitter.

    Args:
        in_sofa_file (str): Path to sofa file.
        legend (list): Legend to populate with receiver dimensions (e.g., Left and Right).
        measurement (int): Measurement index to plot.
        emitter (int): Emitter to plot.
    """
    plt.figure(
        figsize=(15, 5),
        num=(
            "Head-Related Impulse Response at M={0} E={1} for ".format(
                measurement, emitter
            )
            + os.path.basename(in_sofa_file)
        ),
    )
    t, receiver_dimensions, ir = computeHRIR(in_sofa_file, measurement)
    for receiver in np.arange(receiver_dimensions):
        plt.plot(
            t, ir.get_values(indices={"M": measurement, "R": receiver, "E": emitter})
        )
        legend.append("Receiver {0}".format(receiver))
    plt.title(
        "{0}: HRIR at M={1} for emitter {2}".format(
            os.path.basename(in_sofa_file), measurement, emitter
        )
    )
    plt.legend(legend)
    plt.xlabel("$t$ in s")
    plt.ylabel(r"$h(t)$")
    plt.grid()

    return


def plotHRTF(
    in_sofa_file, legend: list, xlim: str, ylim: str, measurement: int, emitter: int
):
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
        plt.figure(
            figsize=(15, 5),
            num=(
                "Head-Related Transfer Function at M={0} E={1} for ".format(
                    measurement, emitter
                )
                + os.path.basename(in_sofa_file)
            ),
        )
        f_axis, HRTF_mag_dB, legend = computeHRTF(in_sofa_file, measurement, emitter)
        plt.semilogx(f_axis, HRTF_mag_dB)
        plt.title(
            "{0}: HRTF at M={1} for emitter {2}".format(
                os.path.basename(in_sofa_file), measurement, emitter
            )
        )

    if sofa_mode_selection == 1:
        in_sofa_files_list = in_sofa_file
        in_sofa_files_list = ", ".join(in_sofa_files_list)
        in_sofa_files_list = in_sofa_files_list.split(",")
        in_sofa_files_list = [file.strip(" ") for file in in_sofa_files_list]
        plt.figure(
            figsize=(15, 5),
            num=str("Left-Channel Head-Related Transfer Function Comparison"),
        )
        for i in in_sofa_files_list:
            legend.append(os.path.basename(i))
            f_axis, HRTF_mag_dB, receiver_legend = computeHRTF(i, measurement, emitter)
            plt.semilogx(f_axis, HRTF_mag_dB)
        plt.title(
            "Left-Channel HRTF Comparison at M={0} for emitter {1}".format(
                measurement, emitter
            )
        )

    ax = plt.gca()
    ax.set_xlim([int(xlim_start), int(xlim_end)])  # Bound the x-axis
    ax.set_ylim([int(ylim_start), int(ylim_end)])  # Bound the y-axis
    plt.grid()  # Horizontal grid line
    plt.grid(which="minor", color="0.9")  # Vertical grid lines
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend(legend)

    return


def viewSOFAGraphs(
    in_sofa_file, xlim: str, ylim: str, measurement: int = 0, emitter: int = 1
):
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
        xlim = "20, 20000"
    if not ylim:
        ylim = "-150, 0"
    if not measurement:
        measurement = 0
    if not emitter:
        emitter = 1

    legend = []

    if sofa_mode_selection == 0:
        in_sofa_file = in_sofa_file[0]

        # plot source coordinates
        sofa_pos_fig = plt.figure(
            figsize=(10, 7),
            num=str("SOFA Source Positions for " + os.path.basename(in_sofa_file)),
        )
        plot_coordinates(in_sofa_file, sofa_pos_fig)

        plotHRIR(in_sofa_file, legend, measurement, emitter)

    plotHRTF(in_sofa_file, legend, xlim, ylim, measurement, emitter)

    plt.show()

    plt.close()
    return


def saveSOFAGraphs(
    in_sofa_file, xlim: str, ylim: str, measurement: int = 0, emitter: int = 1
):
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
        xlim = "20, 20000"
    if not ylim:
        ylim = "-150, 0"
    if not measurement:
        measurement = 0
    if not emitter:
        emitter = 1
    legend = []

    export_directory = filedialog.askdirectory(
        title="Select Save Directory", initialdir=os.path.dirname(in_sofa_file[0])
    )
    if not export_directory:
        errorWindow(error_message="Directory not given.")
        return -1

    if sofa_mode_selection == 0:
        in_sofa_file = in_sofa_file[0]

        export_directory = os.path.join(
            export_directory, os.path.basename(in_sofa_file)
        )
        export_directory = export_directory + "-measurements"
        try:
            os.mkdir(export_directory)
        except FileExistsError:
            pass

        # plot & save source coordinates
        sofa_pos_fig = plt.figure(
            figsize=(10, 7),
            num=str("SOFA Source Positions for " + os.path.basename(in_sofa_file)),
        )
        plot_coordinates(in_sofa_file, sofa_pos_fig)
        plt.savefig(
            os.path.join(
                export_directory,
                (
                    "SOFA_Source_Positions_for_"
                    + os.path.basename(in_sofa_file).replace(" ", "_")
                    + ".png"
                ),
            )
        )

        # plot & save HRIR
        plotHRIR(in_sofa_file, legend, measurement, emitter)
        plt.savefig(
            os.path.join(
                export_directory,
                (
                    "Head-Related_Impulse_Response_at_M={0}_E={1}_for_".format(
                        measurement, emitter
                    )
                    + os.path.basename(in_sofa_file).replace(" ", "_")
                    + ".png"
                ),
            )
        )

    if sofa_mode_selection == 1:
        export_directory = os.path.join(
            export_directory, "hrtf-comparison-measurements"
        )
        try:
            os.mkdir(export_directory)
        except FileExistsError:
            pass

    # plot & save HRTF
    plotHRTF(in_sofa_file, legend, xlim, ylim, measurement, emitter)
    plt.savefig(
        os.path.join(
            export_directory,
            (
                "Left-Channel_HRTF_Comparison_at_M={0}_for_emitter_{1}".format(
                    measurement, emitter
                )
                + ".png"
            ),
        )
    )

    plt.close()

    messageWindow(
        "Successfully exported to:\n" + (os.path.basename(export_directory)),
        title="Export Successful",
        width=400,
        tooltip_text=export_directory,
    )


def renderWithSOFA(
    angle: str,
    elev: str,
    in_source_file: str,
    in_sofa_file: str,
    target_fs: int = 48000,
):
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
        errorWindow("NameError: Missing source file.")
        return
    except TypeError:
        errorWindow("TypeError: Missing source file.")
        return
    except not in_source_file:
        errorWindow("Empty Source File: Missing source file.")
        return
    except type(in_source_file) == None:
        errorWindow("NoneType: Missing source file.")
        return
    else:
        if not in_source_file:
            errorWindow("Missing source file.")
            return

        if not angle:
            angle = 0

        if not elev:
            elev = 0

        # init
        SOFA_HRTF = sofa.Database.open(in_sofa_file)
        sofa_fs_H = SOFA_HRTF.Data.SamplingRate.get_values()[0]
        sofa_positions = SOFA_HRTF.Source.Position.get_values(system="spherical")
        SOFA_H = np.zeros([SOFA_HRTF.Dimensions.N, 2])
        Stereo3D = np.zeros([SOFA_HRTF.Dimensions.N, 2])

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
        [az, az_idx] = find_nearest(sofa_positions[:, 0], angle)
        subpositions = sofa_positions[np.where(sofa_positions[:, 0] == az)]
        [el, sub_idx] = find_nearest(subpositions[:, 1], elev)
        SOFA_H[:, 0] = SOFA_HRTF.Data.IR.get_values(
            indices={"M": az_idx + sub_idx, "R": 0, "E": 0}
        )
        SOFA_H[:, 1] = SOFA_HRTF.Data.IR.get_values(
            indices={"M": az_idx + sub_idx, "R": 1, "E": 0}
        )
        if sofa_fs_H != target_fs:
            # print("---\n** Recalc SR **\nSOFA file SR =", sofa_fs_H, "\nTarget SR =", target_fs)
            # print("---\nOld shape =", SOFA_H.shape)
            SOFA_H = librosa.core.resample(
                SOFA_H.transpose(),
                orig_sr=int(sofa_fs_H),
                target_sr=int(target_fs),
                fix=True,
            ).transpose()
            # print("New shape =", SOFA_H.shape)

        # pick a source (already picked)
        [source_x, fs_x] = sf.read(in_source_file)
        # if it's not mono, make it mono.
        if len(source_x.shape) > 1:
            if source_x.shape[1] > 1:
                source_x = np.mean(source_x, axis=1)
        if fs_x != target_fs:
            # print("** Recalc SR **\nAudio file SR =", fs_x, "\nTarget SR =", target_fs)
            source_x = librosa.core.resample(
                source_x.transpose(),
                orig_sr=int(fs_x),
                target_sr=int(target_fs),
                fix=True,
            ).transpose()

        rend_L = signal.fftconvolve(source_x, SOFA_H[:, 0])
        rend_R = signal.fftconvolve(source_x, SOFA_H[:, 1])
        M = np.max([np.abs(rend_L), np.abs(rend_R)])
        if len(Stereo3D) < len(rend_L):
            diff = len(rend_L) - len(Stereo3D)
            Stereo3D = np.append(Stereo3D, np.zeros([diff, 2]), 0)
        Stereo3D[0 : len(rend_L), 0] += rend_L / M
        Stereo3D[0 : len(rend_R), 1] += rend_R / M

        exportSOFAConvolved(
            in_source_file,
            in_sofa_file,
            angle_label,
            elev_label,
            Stereo3D,
            sofa_positions,
            int(target_fs),
        )

        return Stereo3D


def exportSOFAConvolved(
    in_source_file: str,
    in_sofa_file: str,
    angle_label: int,
    elev_label: int,
    audioContent: np.ndarray,
    sofa_positions: np.ndarray,
    samplerate: int = 48000,
):
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
    sofa_file_location = os.path.join(*in_sofa_file.split("/")[:-1])
    export_directory = filedialog.askdirectory(
        title="Select Save Directory", initialdir=sofa_file_location
    )

    # test for validity of SOFA file. technically, you could pass instead of return -1 here, but if the sofa file doesn't have correct distance information, then it may be malformed in other ways, too.
    try:
        source_distance = sofa_positions[0, 2]
    except:
        errorWindow(error_message="No distance information available.\n")
        return -1

    if export_directory:
        export_filename = (
            str(
                os.path.join(
                    str(export_directory), str(os.path.basename(in_sofa_file)[:-5])
                )
            )
            + "-"
            + str(os.path.basename(in_source_file)[:-4])
            + "-azi_"
            + str(angle_label)
            + "-elev_"
            + str(elev_label)
            + "-export.wav"
        )
        sf.write(export_filename, audioContent, samplerate=samplerate)
        messageWindow(
            message=(
                "Using HRTF set: "
                + str(os.path.basename(in_sofa_file))
                + "\n"
                + "Source distance is: "
                + str(sofa_positions[0, 2])
                + "meters\n\n"
                + "Source: "
                + str(os.path.basename(in_source_file))
                + "\nrendered at azimuth "
                + str(angle_label)
                + " and elevation "
                + str(elev_label)
            ),
            title="SOFA Rendering",
            width=500,
            height=170,
            tooltip_text=str(export_filename),
        )
    if not export_directory:
        errorWindow(error_message="\nNot rendered: Export directory not given.")
        return -1


def spectrogramWindow(audio_file_path: str):
    """
    Creates a window to configure the spectrogram with.

    Args:
        audio_file_path (str): Path to audio file to be passed to the spectrogram function.
    """
    spectrogramConfigWindow = tk.Toplevel(root)
    apply_theme_to_titlebar(spectrogramConfigWindow)
    spectrogramConfigWindow.iconphoto(False, icon_photo)
    centered_window(spectrogramConfigWindow)
    spectrogramConfigWindow.grid_columnconfigure(0, weight=1)
    spectrogramConfigWindow.grid_rowconfigure(0, weight=1)
    spectrogramConfigWindow.minsize(570, 250)
    spectrogramConfigWindow.title("Configure Spectrogram")
    spectrogramConfigWindowContentFrame = tk.Frame(
        spectrogramConfigWindow, borderwidth=10, relief="flat"
    )
    spectrogramConfigWindowContentFrame.grid(row=0, column=0)

    spectrogramConfigTitleLabel = ttk.Label(
        spectrogramConfigWindowContentFrame,
        text="Configure Spectrogram",
        font=("TkDefaultFont", str(parse_font_dict["size"] + 2), "bold"),
    )
    spectrogramConfigTitleLabel.grid(row=0, column=1)
    spectrogramAudioTitleLabel = ttk.Label(
        spectrogramConfigWindowContentFrame, text="Audio file:"
    )
    spectrogramAudioTitleLabel.grid(row=1, column=0)
    spectrogramSelectedAudioLabel = ttk.Label(
        spectrogramConfigWindowContentFrame,
        text=shorten_file_name(os.path.basename(audio_file_path), 24),
    )
    create_tooltip(spectrogramSelectedAudioLabel, text=str(audio_file_path))
    spectrogramSelectedAudioLabel.grid(row=1, column=2)
    startTimeAudioLabel = ttk.Label(
        spectrogramConfigWindowContentFrame, text="Start time (ms):"
    )
    startTimeAudioLabel.grid(row=2, column=0)
    endTimeAudioLabel = ttk.Label(
        spectrogramConfigWindowContentFrame, text="End time (ms):"
    )
    endTimeAudioLabel.grid(row=3, column=0)
    dynamicRangeMinLabel = ttk.Label(
        spectrogramConfigWindowContentFrame, text="Intensity scale min (dB):"
    )
    dynamicRangeMinLabel.grid(row=4, column=0)
    dynamicRangeMaxLabel = ttk.Label(
        spectrogramConfigWindowContentFrame, text="Intensity scale max (dB):"
    )
    dynamicRangeMaxLabel.grid(row=5, column=0)
    plotTitleLabel = ttk.Label(spectrogramConfigWindowContentFrame, text="Plot title:")
    plotTitleLabel.grid(row=6, column=0)

    startTimeStringVar = tk.StringVar()
    endTimeStringVar = tk.StringVar()
    dynamicRangeMinStringVar = tk.StringVar()
    dynamicRangeMaxStringVar = tk.StringVar()
    plotTitleStringVar = tk.StringVar()

    startTimeAudioEntry = ttk.Entry(
        spectrogramConfigWindowContentFrame, textvariable=startTimeStringVar
    )
    startTimeAudioEntry.grid(row=2, column=2)
    endTimeAudioEntry = ttk.Entry(
        spectrogramConfigWindowContentFrame, textvariable=endTimeStringVar
    )
    endTimeAudioEntry.grid(row=3, column=2)
    dynamicRangeMinEntry = ttk.Entry(
        spectrogramConfigWindowContentFrame, textvariable=dynamicRangeMinStringVar
    )
    dynamicRangeMinEntry.grid(row=4, column=2)
    dynamicRangeMaxEntry = ttk.Entry(
        spectrogramConfigWindowContentFrame, textvariable=dynamicRangeMaxStringVar
    )
    dynamicRangeMaxEntry.grid(row=5, column=2)
    plotTitleEntry = ttk.Entry(
        spectrogramConfigWindowContentFrame, textvariable=plotTitleStringVar
    )
    plotTitleEntry.grid(row=6, column=2)

    viewSpectrogramButton = ttk.Button(
        spectrogramConfigWindowContentFrame,
        text="View spectrogram",
        style="my.TButton",
        command=lambda: spectrogram(
            audio_file_path,
            startTimeStringVar.get(),
            endTimeStringVar.get(),
            dynamicRangeMinStringVar.get(),
            dynamicRangeMaxStringVar.get(),
            plotTitleStringVar.get(),
        ),
    )
    viewSpectrogramButton.grid(row=7, column=1)


def spectrogram(
    audio_file_path: str,
    start_ms: str,
    end_ms: str,
    dynamic_range_min: str,
    dynamic_range_max: str,
    plot_title: str,
):
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
        start_ms = "0"
    if not end_ms:
        end_ms = str((len(samples) / sr) * 1000)

    start_in_samples = (float(start_ms) / 1000) * sr
    end_in_samples = (float(end_ms) / 1000) * sr

    if start_in_samples >= end_in_samples:
        errorWindow(
            error_message="End time cannot be equal to or less than start time.",
            width=400,
        )
        return

    start_in_samples = int(start_in_samples)
    end_in_samples = int(end_in_samples)
    rebound_samples = samples[start_in_samples:end_in_samples]

    f, t, spectrogram = signal.spectrogram(rebound_samples, sr)

    fig, ax = plt.subplots()
    if dynamic_range_min and dynamic_range_max:
        p = ax.pcolormesh(
            t,
            f,
            10 * np.log10(spectrogram),
            vmin=int(dynamic_range_min),
            vmax=int(dynamic_range_max),
            shading="auto",
        )
    elif not dynamic_range_min and dynamic_range_max:
        p = ax.pcolormesh(
            t,
            f,
            10 * np.log10(spectrogram),
            vmax=int(dynamic_range_max),
            shading="auto",
        )
    elif dynamic_range_min and not dynamic_range_max:
        p = ax.pcolormesh(
            t,
            f,
            10 * np.log10(spectrogram),
            vmin=int(dynamic_range_min),
            shading="auto",
        )
    else:
        p = ax.pcolormesh(t, f, 10 * np.log10(spectrogram), shading="auto")

    ax.set_ylim(1, int(sr / 2))
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")

    fig.colorbar(p, label="Intensity (dB)")
    fig.canvas.manager.set_window_title(
        str("Spectrogram for " + os.path.basename(audio_file_path))
    )

    plt.title(str(plot_title))
    plt.show()


def callback_url(url: str):
    webbrowser.open_new(url)


def clearWidgets(frame_to_clear):
    """
    Clears given frame.

    Args:
        frame_to_clear (frame): tk.Frame to clear
    """    
    for widget in frame_to_clear.winfo_children():
        widget.destroy()


def showPreferencesWindow(): # unused, idk what preferences i'd even include in a program like this, and if i do include preferences, then i need to store them somewhere and that seems like a headache
    preferencesWindow = tk.Toplevel(root)
    apply_theme_to_titlebar(preferencesWindow)
    preferencesWindow.iconphoto(False, icon_photo)
    preferencesWindow.grid_columnconfigure(0, weight=1)
    preferencesWindow.grid_rowconfigure(0, weight=1)
    preferencesWindow.minsize(400, 450)
    preferencesWindow.title('Preferences')
    preferencesWindowContentFrame = tk.Frame(preferencesWindow, borderwidth=10, relief="flat")
    preferencesWindowContentFrame.grid(row=0, column=0)
    centered_window(preferencesWindow)


def createHelpWindow():
    global tutorialWindow
    global tutorialWindowContentFrame
    tutorialWindow = tk.Toplevel(root)
    apply_theme_to_titlebar(tutorialWindow)
    tutorialWindow.iconphoto(False, icon_photo)
    tutorialWindow.grid_columnconfigure(0, weight=1)
    tutorialWindow.grid_rowconfigure(0, weight=1)
    tutorialWindow.minsize(1250, 750)
    tutorialWindow.title("Help")
    tutorialWindowContentFrame = tk.Frame(tutorialWindow, borderwidth=10, relief="flat")
    tutorialWindowContentFrame.grid(row=0, column=0)
    centered_window(tutorialWindow)
    hrtfHelpPage()


def hrtfHelpPage():
    clearWidgets(tutorialWindowContentFrame)

    hrtfTitleTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text="HRTF Functions\n",
        font=("TkDefaultFont", str(parse_font_dict["size"] + 4), "bold"),
    )
    hrtfTitleTutorialLabel.grid(row=0, column=1)

    selectHRTFTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Select HRTF file (.wav)..."\nPresents dialogue box for selecting HRTF.\nOnly takes .wav files. Expects an IR.\n',
    )
    selectHRTFTutorialLabel.grid(row=1, column=0)
    getHRTFFileDataTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Get HRTF file data"\nPresents info about HRTF file, including:\n- Sample rate\n- Data dimension (num samples, num channels)\n- Ability to play loaded HRTF file.\n',
    )
    getHRTFFileDataTutorialLabel.grid(row=2, column=0)
    hrtfTimeDomainVisualizationTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"HRTF time domain visualization"\nTime domain plot of loaded HRTF.\n',
    )
    hrtfTimeDomainVisualizationTutorialLabel.grid(row=3, column=0)
    hrtfFrequencyDomainVisualizationTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"HRTF frequency domain visualization"\nFrequency domain plot of loaded HRTF.\n',
    )
    hrtfFrequencyDomainVisualizationTutorialLabel.grid(row=4, column=0)

    selectSourceFileTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Select source file (.wav)..."\nPresents dialogue box for selecting a source file.\nOnly takes .wav files.\nUsed for convolving with loaded HRTF.\nAlso used for convolving with loaded SOFA file.\n',
    )
    selectSourceFileTutorialLabel.grid(row=1, column=2)
    getSourceFileDataTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Get source file data"\nPresents info about source file, including:\n- Sample rate\n- Data dimension (num samples, num channels)\n- Ability to play loaded selected file.\n',
    )
    getSourceFileDataTutorialLabel.grid(row=2, column=2)
    spectrogramTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"View spectrogram..."\nOpens a menu for configuring and\n viewing a spectrogram of the loaded source file.\n',
    )
    spectrogramTutorialLabel.grid(row=3, column=2)
    stereoToMonoTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Source file stereo -> mono"\nDownmixes stereo file into mono\nfor convenience with convolving.\n',
    )
    stereoToMonoTutorialLabel.grid(row=4, column=2)

    resampleTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Resample"\nResamples source file to match sample rate of loaded HRTF file.\nResampled source file is held in memory, not exported.\nSource File Stereo -> Mono MUST be pressed first!\n',
    )
    resampleTutorialLabel.grid(row=5, column=1)
    timeDomainConvolveTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Time domain convolve"\nTime domain convolves loaded source file with loaded HRTF file.\nSource file should either match HRTF file sample rate,\nor have been resampled with the button above.\nConvolved file is held in memory, not exported.\n',
    )
    timeDomainConvolveTutorialLabel.grid(row=6, column=1)
    exportConvolvedTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Export convolved..."\nExports the time domain convolved file loaded in memory.\nFile naming convention is:\n[Source File Name]-[HRTF File Name]-export.wav\n',
    )
    exportConvolvedTutorialLabel.grid(row=7, column=1)

    nextButton = ttk.Button(
        tutorialWindowContentFrame,
        text="Next (SOFA Help) ->",
        style="my.TButton",
        command=lambda: sofaHelpPage(),
    )
    nextButton.grid(row=8, column=2, sticky="E")


def sofaHelpPage():
    clearWidgets(tutorialWindowContentFrame)
    sofaTitleTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text="SOFA Functions\n",
        font=("TkDefaultFont", str(parse_font_dict["size"] + 4), "bold"),
    )
    sofaTitleTutorialLabel.grid(row=0, column=1)

    selectSOFAFileTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Select SOFA file(s) (.sofa)..."\nPresents dialogue box for selecting .SOFA file(s).\n',
    )
    selectSOFAFileTutorialLabel.grid(row=1, column=1)
    getSOFAFileMetadataTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Get SOFA file metadata"\nPresents metadata embedded in the loaded SOFA file.\nFollows SOFA convention.\n',
    )
    getSOFAFileMetadataTutorialLabel.grid(row=2, column=0)
    sofaMeasurementTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Measurement index"\nChoose the measurement index to be used when viewing plot data.\nCheck .SOFA file dimensions for measurement indices.\n',
    )
    sofaMeasurementTutorialLabel.grid(row=3, column=0)
    frequencyXLimTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Frequency range (Hz)"\nConfigurable range for x-axis of .SOFA file plot.\nDefaults to [20, 20000].\n',
    )
    frequencyXLimTutorialLabel.grid(row=4, column=0)
    desiredAzimuthTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Desired azimuth (in deg)"\nEnter azimuth for rendering with source file. Defaults to 0deg.\nSelectable azimuth for viewing .SOFA file plot, and for rendering.\n',
    )
    desiredAzimuthTutorialLabel.grid(row=5, column=0)

    getSOFAFileDimensionsTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Get SOFA file dimensions"\nPresents info about the SOFA convention dimensions\nwithin the .SOFA file.\n',
    )
    getSOFAFileDimensionsTutorialLabel.grid(row=2, column=2)
    sofaEmitterTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Emitter"\nChoose the emitter to be used when viewing plot data.\nCheck .SOFA file dimensions for emitters.\n',
    )
    sofaEmitterTutorialLabel.grid(row=3, column=2)
    magnitudeYLimTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Magnitude (dB)"\nConfigurable range for y-axis of .SOFA file plot.\nDefaults to [-150, 0].\n',
    )
    magnitudeYLimTutorialLabel.grid(row=4, column=2)
    desiredElevationTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Desired elevation (in deg)"\nEnter elevation for rendering with source file. Defaults to 0deg.\nSelectable elevation for viewing .SOFA file plot, and for rendering.\n',
    )
    desiredElevationTutorialLabel.grid(row=5, column=2)

    viewSOFAFileTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"View SOFA file/View SOFA HRTF"\nTakes the above selected values\nand presents a 3D view of the .SOFA file,\n in addition to individual measurements\nfrom the .SOFA file.\nIf multiple .SOFA files are selected,\n the only graph displayed will be a layered HRTF graph.\n',
    )
    viewSOFAFileTutorialLabel.grid(row=6, column=0)
    saveSOFAFileTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Save all SOFA graphs.../Save SOFA HRTF..."\nSaves graphs/plots for source positions,\nhead-related impulse response, and head-related transfer function\nfor the provided azimuth, elevation, emitter, and measurement index to\nthe provided directory.\nIf multiple .SOFA files are selected,\n the only graph saved will be a layered HRTF graph.\n',
    )
    saveSOFAFileTutorialLabel.grid(row=6, column=2)
    renderSOFATutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text='"Render source with SOFA file..."\nConvolves the source file with\nthe desired values in the .SOFA file.\nDisabled if multiple .SOFA files are selected.\n',
    )
    renderSOFATutorialLabel.grid(row=7, column=1)

    prevButton = ttk.Button(
        tutorialWindowContentFrame,
        text="<- Previous (HRTF Help)",
        style="my.TButton",
        command=lambda: hrtfHelpPage(),
    )
    prevButton.grid(row=8, column=0, sticky="W")

    nextButton = ttk.Button(
        tutorialWindowContentFrame,
        text="Next (General Help) ->",
        style="my.TButton",
        command=lambda: generalHelpPage(),
    )
    nextButton.grid(row=8, column=2, sticky="E")


def generalHelpPage():
    clearWidgets(tutorialWindowContentFrame)
    generalTitleTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text="General Help\n",
        font=("TkDefaultFont", str(parse_font_dict["size"] + 4), "bold"),
    )
    generalTitleTutorialLabel.grid(row=0, column=1)

    whatIsTitleTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text="What is this program?",
        font=("TkDefaultFont", str(parse_font_dict["size"] + 2), "bold"),
    )
    whatIsTitleTutorialLabel.grid(row=1, column=0)
    whatIsDescTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text="""
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
    at any captured azimuth and elevation within the SOFA file, and more.""",
    )
    whatIsDescTutorialLabel.grid(row=2, column=0)
    howToTitleTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text="How do I use this program?",
        font=("TkDefaultFont", str(parse_font_dict["size"] + 2), "bold"),
    )
    howToTitleTutorialLabel.grid(row=1, column=2)
    howToDescTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text="""
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
    """,
    )
    howToDescTutorialLabel.grid(row=2, column=2)

    commonHelpTitleTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text="Tips/Tricks",
        font=("TkDefaultFont", str(parse_font_dict["size"] + 2), "bold"),
    )
    commonHelpTitleTutorialLabel.grid(row=3, column=0)
    commonHelpDescTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text="""
    * 'Source File Stereo -> Mono' must ALWAYS be pressed before convolving with HRTF!
    * You can use 'Tab' to select a subsequent text box.
    * The default values for SOFA functions are as follows:
        - Measurement Index: 0
        - Emitter: 1
        - Frequency Range (Hz): 20, 20000
        - Magnitude (dB): -150, 0
        - Azimuth (deg): 0
        - Elevation (deg): 0
    """,
        justify="left",
    )
    commonHelpDescTutorialLabel.grid(row=4, column=0, rowspan=8)

    if darkdetect.isDark():
        light_or_dark_blue = "lightblue"
    else:
        light_or_dark_blue = "darkblue"

    feedbackTitleTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text="Contact/Feedback",
        font=("TkDefaultFont", str(parse_font_dict["size"] + 2), "bold"),
    )
    feedbackTitleTutorialLabel.grid(row=3, column=2)
    feedbackDescTutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text="https://hytheaway.github.io/contact.html",
        fg=light_or_dark_blue,
        cursor="hand2",
    )
    feedbackDescTutorialLabel.grid(row=4, column=2)
    feedbackDescTutorialLabel.bind(
        "<Button-1>", lambda e: callback_url("https://hytheaway.github.io/contact.html")
    )
    feedbackDesc2TutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text="https://github.com/hytheaway",
        fg=light_or_dark_blue,
        cursor="hand2",
    )
    feedbackDesc2TutorialLabel.grid(row=5, column=2)
    feedbackDesc2TutorialLabel.bind(
        "<Button-1>", lambda e: callback_url("https://github.com/hytheaway")
    )
    feedbackDesc3TutorialLabel = tk.Label(
        tutorialWindowContentFrame,
        text="hytheaway@gmail.com",
        fg=light_or_dark_blue,
        cursor="hand2",
    )
    feedbackDesc3TutorialLabel.grid(row=6, column=2)
    feedbackDesc3TutorialLabel.bind(
        "<Button-1>", lambda e: callback_url("mailto:hytheaway@gmail.com")
    )

    prevButton = ttk.Button(
        tutorialWindowContentFrame,
        text="<- Previous (SOFA Help)",
        style="my.TButton",
        command=lambda: sofaHelpPage(),
    )
    prevButton.grid(row=20, column=0, sticky="W")

def quit_function():
    """
    Quit function that properly closes out of pygame and the python script, which prevents a segfault when this project is compiled by nuitka.
    """    
    pygame.quit()
    sys.exit()

def apply_theme_to_titlebar(
    root,
):  # https://github.com/rdbende/Sun-Valley-ttk-theme/tree/main
    if sys.platform == 'win32':
        version = sys.getwindowsversion()

        if version.major == 10 and version.build >= 22000:
            # Set the title bar color to the background color on Windows 11 for better appearance
            pywinstyles.change_header_color(
                root, "#1c1c1c" if sv_ttk.get_theme() == "dark" else "#fafafa"
            )
        elif version.major == 10:
            pywinstyles.change_header_color(
                root, "dark" if sv_ttk.get_theme() == "dark" else "normal"
            )
            # A hacky way to update the title bar's color on Windows 10 (it doesn't update instantly like on Windows 11)
            root.wm_attributes("-alpha", 0.99)
            root.wm_attributes("-alpha", 1)
    else:
        return


root = tk.Tk()
root.minsize(565, 960)
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

# print(root.tk.call('tk', 'windowingsystem'))

# menu bar stuff
root.option_add('*tearOff', False)

root_menubar = tk.Menu(root)
root['menu'] = root_menubar

# File menu
root_menu_file = tk.Menu(root_menubar)
root_menubar.add_cascade(menu=root_menu_file, label='File')

root_menu_file_hrtf = tk.Menu(root_menu_file)
root_menu_file.add_cascade(menu=root_menu_file_hrtf, label='HRTF')
root_menu_file_hrtf.add_command(label='Load HRTF file (.wav)...',
                                command=lambda: selectHRTFFile())
root_menu_file_hrtf.add_separator()
root_menu_file_hrtf.add_command(label='Get HRTF file data',
                                command=lambda: getHRTFFileData(hrtf_file, HRIR),
                                state='disabled')
root_menu_file_hrtf.add_command(label='HRTF time domain visualization',
                                command=lambda: timeDomainVisualHRTF(hrtf_file, HRIR),
                                state='disabled')
root_menu_file_hrtf.add_command(label='HRTF frequency domain visualization',
                                command=lambda: freqDomainVisualHRTF(hrtf_file),
                                state='disabled')

root_menu_file_source = tk.Menu(root_menu_file)
root_menu_file.add_cascade(menu=root_menu_file_source, label='Source')
root_menu_file_source.add_command(label='Load source file (.wav)...',
                                  command=lambda: selectSourceFile())
root_menu_file_source.add_separator()
root_menu_file_source.add_command(label='Get source file data',
                                  command=lambda: getSourceFileData(source_file),
                                  state='disabled')
root_menu_file_source.add_command(label='View spectrogram...',
                                  command=lambda: spectrogramWindow(source_file),
                                  state='disabled')
root_menu_file_source.add_command(label='Source file stereo -> mono',
                                  command=lambda: stereoToMono(sig),
                                  state='disabled')

root_menu_file_sofa = tk.Menu(root_menu_file)
root_menu_file.add_cascade(menu=root_menu_file_sofa, label='SOFA')
root_menu_file_sofa.add_command(label='Load SOFA file (.sofa)...',
                                command=lambda: selectSOFAFile())
root_menu_file_sofa.add_separator()
root_menu_file_sofa.add_command(label='Get SOFA file metadata',
                                command=lambda: getSOFAFileMetadata(sofa_file_path_list[0]),
                                state='disabled')
root_menu_file_sofa.add_command(label='Get SOFA file dimensions',
                                command=lambda: getSOFAFileDimensions(sofa_file_path_list[0]),
                                state='disabled')
root_menu_file_sofa.add_separator()
root_menu_file_sofa.add_command(label='View SOFA file',
                                command=lambda: viewSOFAGraphs(
                                                    sofa_file_path_list,
                                                    freqXLimStringVar.get(),
                                                    magYLimStringVar.get(),
                                                    sofaMeasurementStringVar.get(),
                                                    sofaEmitterStringVar.get(),
                                                ),
                                state='disabled')
root_menu_file_sofa.add_command(label='Save all SOFA graphs...',
                                command=lambda: saveSOFAGraphs(
                                                    sofa_file_path_list,
                                                    freqXLimStringVar.get(),
                                                    magYLimStringVar.get(),
                                                    sofaMeasurementStringVar.get(),
                                                    sofaEmitterStringVar.get(),
                                                ),
                                state='disabled')
root_menu_file_sofa.add_separator()
root_menu_file_sofa.add_command(label='Render source with SOFA file...',
                                command=lambda: renderWithSOFA(
                                                    azimuthStringVar.get(),
                                                    elevationStringVar.get(),
                                                    source_file,
                                                    sofa_file_path_list[0],
                                                ),
                                state='disabled')


if sys.platform == "darwin": # macOS
    # Shortcuts for file menu
    root_menu_file_hrtf.entryconfigure('Load HRTF file (.wav)...', accelerator='Control+H')
    root.bind('<Control-h>', lambda x:selectHRTFFile())
    
    root_menu_file_source.entryconfigure('Load source file (.wav)...', accelerator='Control+R')
    root.bind('<Control-r>', lambda x:selectSourceFile())
    
    root_menu_file_sofa.entryconfigure('Load SOFA file (.sofa)...', accelerator='Control+F')
    root.bind('<Control-f>', lambda x:selectSOFAFile())
    
    # macOS window menu
    window_menu = tk.Menu(root_menubar, name='window')
    root_menubar.add_cascade(menu=window_menu, label='Window')
    
    # macOS help menu
    help_menu = tk.Menu(root_menubar, name='help')
    root_menubar.add_cascade(menu=help_menu, label='Help')
    root.createcommand('tk::mac::ShowHelp', lambda:createHelpWindow())
    
    pass

if sys.platform == 'win32':
    root_menu_file_hrtf.entryconfigure('Load HRTF file (.wav)...', accelerator='Control+H')
    root.bind('<Control-h>', lambda x:selectHRTFFile())
    
    root_menu_file_source.entryconfigure('Load source file (.wav)...', accelerator='Control+R')
    root.bind('<Control-r>', lambda x:selectSourceFile())
    
    root_menu_file_sofa.entryconfigure('Load SOFA file (.sofa)...', accelerator='Control+F')
    root.bind('<Control-f>', lambda x:selectSOFAFile())
    
    # windows system menu
    sysmenu = tk.Menu(root_menubar, name='system')
    root_menubar.add_cascade(menu=sysmenu)
    pass

centered_window(root)
try:
    icon_photo = tk.PhotoImage(file="share/happyday.png")
except:
    icon_photo_data = "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"
    icon_photo = tk.PhotoImage(data=b64decode(icon_photo_data))
root.iconphoto(False, icon_photo)
root.title("GGH&SF")
default_font = font.nametofont("TkDefaultFont")
parse_font_dict = default_font.actual()

icon_label = tk.Label(root, image=icon_photo)
icon_label.place(x=0, y=0, width=120, height=120, relwidth=0.5, relheight=0.5)

rootFrame = tk.Frame(root, borderwidth=10, relief="flat")
rootFrame.grid(row=0, column=0)

titleLabel = ttk.Label(
    rootFrame,
    text="Garrett's Great\nHRTF Functions\n",
    justify="center",
    font=("TkDefaultFont", str(parse_font_dict["size"] + 2), "bold"),
)
titleLabel.grid(row=0, column=0, columnspan=3)

topSectionFrame = tk.Frame(rootFrame, borderwidth=10, relief="ridge")
topSectionFrame.grid(row=1, column=0, columnspan=3)

hrtfSourceSelectionFrame = tk.Frame(topSectionFrame, borderwidth=5, relief="flat")
hrtfSourceSelectionFrame.grid(row=0, column=0, columnspan=3)

hrtfFrame = tk.Frame(hrtfSourceSelectionFrame, borderwidth=10, relief="flat")
hrtfFrame.grid(row=0, column=0)

selectHRTFFileButton = ttk.Button(
    hrtfFrame,
    text="Select HRTF file (.wav)...",
    style="my.TButton",
    command=lambda: selectHRTFFile(),
)
selectHRTFFileButton.grid(row=0, column=0)
selectHRTFFileLabel = ttk.Label(
    hrtfFrame, text="HRTF file:\n", justify="center", wraplength=120
)
selectHRTFFileLabel.grid(row=1, column=0)
getHRTFFileDataButton = ttk.Button(
    hrtfFrame,
    text="Get HRTF file data",
    style="my.TButton",
    state="disabled",
    command=lambda: getHRTFFileData(hrtf_file, HRIR),
)
getHRTFFileDataButton.grid(row=3, column=0)
timeDomainVisualHRTFButton = ttk.Button(
    hrtfFrame,
    text="HRTF time domain visualization",
    style="my.TButton",
    state="disabled",
    command=lambda: timeDomainVisualHRTF(hrtf_file, HRIR),
)
timeDomainVisualHRTFButton.grid(row=4, column=0)
freqDomainVisualHRTFButton = ttk.Button(
    hrtfFrame,
    text="HRTF frequency domain visualization",
    style="my.TButton",
    state="disabled",
    command=lambda: freqDomainVisualHRTF(hrtf_file),
)
freqDomainVisualHRTFButton.grid(row=5, column=0)

sourceFrame = tk.Frame(hrtfSourceSelectionFrame, borderwidth=10, relief="flat")
sourceFrame.grid(row=0, column=2)
selectSourceFileButton = ttk.Button(
    sourceFrame,
    text="Select source file (.wav)...",
    style="my.TButton",
    command=lambda: selectSourceFile(),
)
selectSourceFileLabel = ttk.Label(
    sourceFrame, text="Source file:\n", justify="center", wraplength=120
)
selectSourceFileButton.grid(row=0, column=2)
selectSourceFileLabel.grid(row=1, column=2)
getSourceFileDataButton = ttk.Button(
    sourceFrame,
    text="Get source file data",
    style="my.TButton",
    state="disabled",
    command=lambda: getSourceFileData(source_file),
)
getSourceFileDataButton.grid(row=2, column=2)
spectrogramButton = ttk.Button(
    sourceFrame,
    text="View spectrogram...",
    style="my.TButton",
    state="disabled",
    command=lambda: spectrogramWindow(source_file),
)
spectrogramButton.grid(row=3, column=2)
stereoToMonoButton = ttk.Button(
    sourceFrame,
    text="Source file stereo -> mono",
    style="my.TButton",
    state="disabled",
    command=lambda: stereoToMono(sig),
)
stereoToMonoButton.grid(row=4, column=2)

hrtfOperationsFrame = tk.Frame(topSectionFrame, borderwidth=10, relief="flat")
hrtfOperationsFrame.grid(row=1, column=1)

resampleButton = ttk.Button(
    hrtfOperationsFrame,
    text="Resample",
    style="my.TButton",
    state="disabled",
    command=lambda: fs_resample(sig_mono, fs_s, HRIR, fs_H),
)
resampleButton.grid(row=1, column=1)
timeDomainConvolveButton = ttk.Button(
    hrtfOperationsFrame,
    text="Time domain convolve",
    style="my.TButton",
    state="disabled",
    command=lambda: timeDomainConvolve(sig_mono, HRIR),
)
timeDomainConvolveButton.grid(row=2, column=1)
exportConvolvedButton = ttk.Button(
    hrtfOperationsFrame,
    text="Export convolved...",
    style="my.TButton",
    state="disabled",
    command=lambda: exportConvolved(Bin_Mix, fs_s, source_file, hrtf_file),
)
exportConvolvedButton.grid(row=3, column=1)

sectionalLabel = ttk.Label(rootFrame, text="\n")
sectionalLabel.grid(row=2, column=0, columnspan=3)

sofaLabel = ttk.Label(
    rootFrame,
    text="Garrett's Great\nSOFA Functions\n",
    justify="center",
    font=("TkDefaultFont", str(parse_font_dict["size"] + 2), "bold"),
)
sofaLabel.grid(row=3, column=0, columnspan=3)

bottomSectionFrame = tk.Frame(rootFrame, borderwidth=10, relief="ridge")
bottomSectionFrame.grid(row=4, column=0, columnspan=3)

selectSOFAFileButton = ttk.Button(
    bottomSectionFrame,
    text="Select SOFA file(s) (.sofa)...",
    style="my.TButton",
    command=lambda: selectSOFAFile(),
)
selectSOFAFileButton.grid(row=0, column=0, columnspan=3)
selectSOFAFileLabel = ttk.Label(
    bottomSectionFrame, text="SOFA file:\n", justify="center", wraplength=240
)
selectSOFAFileLabel.grid(row=1, column=0, columnspan=3)

sofaMeasurementStringVar = tk.StringVar()
sofaEmitterStringVar = tk.StringVar()
freqXLimStringVar = tk.StringVar()
magYLimStringVar = tk.StringVar()
azimuthStringVar = tk.StringVar()
elevationStringVar = tk.StringVar()

bottomLeftFrame = tk.Frame(bottomSectionFrame, borderwidth=10, relief="flat")
bottomLeftFrame.grid(row=2, column=0)
bottomRightFrame = tk.Frame(bottomSectionFrame, borderwidth=10, relief="flat")
bottomRightFrame.grid(row=2, column=2)

getSOFAFileMetadataButton = ttk.Button(
    bottomLeftFrame,
    text="Get SOFA file metadata",
    style="my.TButton",
    state="disabled",
    command=lambda: getSOFAFileMetadata(sofa_file_path_list[0]),
)
getSOFAFileMetadataButton.grid(row=0, column=0)

getSOFAFileDimensionsButton = ttk.Button(
    bottomRightFrame,
    text="Get SOFA file dimensions",
    style="my.TButton",
    state="disabled",
    command=lambda: getSOFAFileDimensions(sofa_file_path_list[0]),
)
getSOFAFileDimensionsButton.grid(row=0, column=0)

bottomLeftSeparator = ttk.Label(bottomLeftFrame, text='', justify='center')
bottomLeftSeparator.grid(row=1, column=0)

bottomRightSeparator = ttk.Label(bottomRightFrame, text='', justify='center')
bottomRightSeparator.grid(row=1, column=0)

sofaMeasurementTextBox = ttk.Entry(
    bottomLeftFrame, state="disabled", width=5, textvariable=sofaMeasurementStringVar
)
sofaMeasurementTextBox.grid(row=2, column=0)
sofaMeasurementLabel = ttk.Label(
    bottomLeftFrame, text="Measurement index\n(default: 0)\n", justify="center"
)
sofaMeasurementLabel.grid(row=3, column=0)
sofaEmitterTextBox = ttk.Entry(
    bottomRightFrame, state="disabled", width=5, textvariable=sofaEmitterStringVar
)
sofaEmitterTextBox.grid(row=2, column=0)
sofaEmitterLabel = ttk.Label(
    bottomRightFrame, text="Emitter\n(default: 1)\n", justify="center"
)
sofaEmitterLabel.grid(row=3, column=0)

frequencyXLimTextBox = ttk.Entry(
    bottomLeftFrame, state="disabled", width=15, textvariable=freqXLimStringVar
)
frequencyXLimTextBox.grid(row=4, column=0)

frequencyXLimLabel = ttk.Label(
    bottomLeftFrame, text="Frequency range (Hz)\n[start, end]\n", justify="center"
)
frequencyXLimLabel.grid(row=5, column=0)

magnitudeYLimTextBox = ttk.Entry(
    bottomRightFrame, state="disabled", width=15, textvariable=magYLimStringVar
)
magnitudeYLimTextBox.grid(row=4, column=0)

magnitudeYLimLabel = ttk.Label(
    bottomRightFrame, text="Magnitude (dB)\n[start, end]\n", justify="center"
)
magnitudeYLimLabel.grid(row=5, column=0)

sofaViewButton = ttk.Button(
    bottomLeftFrame,
    text="View SOFA file",
    style="my.TButton",
    state="disabled",
    command=lambda: viewSOFAGraphs(
        sofa_file_path_list,
        freqXLimStringVar.get(),
        magYLimStringVar.get(),
        sofaMeasurementStringVar.get(),
        sofaEmitterStringVar.get(),
    ),
)
# sofaViewButton.grid(row=4, column=0, columnspan=2)
sofaViewButton.grid(row=6, column=0)

sofaSaveButton = ttk.Button(
    bottomRightFrame,
    text="Save all SOFA graphs...",
    style="my.TButton",
    state="disabled",
    command=lambda: saveSOFAGraphs(
        sofa_file_path_list,
        freqXLimStringVar.get(),
        magYLimStringVar.get(),
        sofaMeasurementStringVar.get(),
        sofaEmitterStringVar.get(),
    ),
)
# sofaSaveButton.grid(row=4, column=1, columnspan=2)
sofaSaveButton.grid(row=6, column=0)

bottomLeftSeparator2 = ttk.Label(bottomLeftFrame, text='', justify='center')
bottomLeftSeparator2.grid(row=7, column=0)

bottomRightSeparator2 = ttk.Label(bottomRightFrame, text='', justify='center')
bottomRightSeparator2.grid(row=7, column=0)

azimuthTextBox = ttk.Entry(
    bottomLeftFrame, state="disabled", width=5, textvariable=azimuthStringVar
)
azimuthTextBox.grid(row=8, column=0)
azimuthLabel = ttk.Label(
    bottomLeftFrame, text="Desired azimuth (in deg)", justify="center"
)
azimuthLabel.grid(row=9, column=0)
elevationTextBox = ttk.Entry(
    bottomRightFrame, state="disabled", width=5, textvariable=elevationStringVar
)
elevationTextBox.grid(row=8, column=0)
elevationLabel = ttk.Label(
    bottomRightFrame, text="Desired elevation (in deg)", justify="center"
)
elevationLabel.grid(row=9, column=0)

sofaRenderButton = ttk.Button(
    bottomSectionFrame,
    text="Render source with SOFA file...",
    style="my.TButton",
    state="disabled",
    command=lambda: renderWithSOFA(
        azimuthStringVar.get(),
        elevationStringVar.get(),
        source_file,
        sofa_file_path_list[0],
    ),
)
sofaRenderButton.grid(row=5, column=0, columnspan=3)

tutorialButton = ttk.Button(
    rootFrame, text="Help", style="my.TButton", command=lambda: createHelpWindow()
)
tutorialButton.grid(row=6, column=0, sticky="W")

quitButton = ttk.Button(
    rootFrame, text="Quit", style="my.TButton", command=lambda: quit_function()
)
quitButton.grid(row=6, column=2, sticky="E")

# prevents the window from appearing at the bottom of the stack
root.focus_force()

root.protocol("WM_DELETE_WINDOW", lambda: sys.exit())

if sys.platform != 'win32':
    ttkStyles = ttk.Style()
    ttkStyles.configure("my.TButton", font=(default_font, 12))

sv_ttk.set_theme(darkdetect.theme())

if sys.platform == 'win32':
    ttkStyles = ttk.Style()
    ttkStyles.configure("my.TButton", font=("SunValleyBodyFont", 9))
    apply_theme_to_titlebar(root) # yes i know this is redundant
    pass

root.mainloop()
