# ggh&sf
garrett's great hrtf (& sofa) functions

the best (and only) dedicated tool for analyzing both hrtf files and sofa files, while also containing the ability to convolve them and export the convolutions! does it ever get better than this?

and it's lightweight to boot! that is, if you consider a 100+ MB python file that takes at least 10 seconds to boot up, "lightweight".

speaking of which: **on macOS, the app will seem like it has crashed for the first 10 seconds or so. i promise you it hasn't, so please be patient.**
this is because pyinstaller trashes cached files when the program is closed, forcing matplotlib to recache all fonts on each bootup. i don't like it, but until tk/tcl's graphics library gets secondary threads working on macOS, i can't do much about it. 

as another note, i have only tested (and will only support) windows 11. windows 10 *should* work, since the run-time .dlls from visual studio 2015 are included in windows 10 onward, but i have not tested this. if you intend on running this on an older version of windows (vista to 8.1), ensure that you have the run-time
.dlls for visual studio 2015 (more info here: [universal crt](https://devblogs.microsoft.com/cppblog/introducing-the-universal-crt)). you should be able to get this from downloading the [windows 10 sdk](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/).

![gghsf_main](https://github.com/user-attachments/assets/d64cbff4-215c-4469-9119-f5f56db2ea23)
