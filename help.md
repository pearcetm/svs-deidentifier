# How do I use this to de-identify whole slide images?

### There are two modes of use:
- **Copy-and-delabel**: Leaves the original file unmodified, only removing the label and macro images from the new copy. De-labeling is extremely fast, but copying large WSI files can be quite slow. 99.9% of the time that it takes to use this mode is in the file copy operation.  

- **Modify in place**: Delete the label and macro images from the **original file**. Very fast but ***cannot be undone!***. Use with ***extreme care*** especially if working in an environment with clinical images.

In both modes, you start by selecting the .svs files to process. These files can be selected directly, or by loading a .csv file (see below for more information). Once your files are set up correctly, hit the button to perform deidentification.

### Use the settings page to customize your preferences
Certain functions can work in multiple ways, and the settings page lets you choose which mode you prefer.
- **File dialog type**: Choose whether you want the operating system style of file picker, or the cross-platform Qt version. The right choice may depend on your specific operating system or computer configuration. For example, on macOS Catalina, the native dialog can be problematic. When testing the dialog choices, try opening the dialog more than once to make sure the dialog behavior is consistent.  

- **File dialog directory**: The app can either start in the same place every time while looking for files, or remember the last directory you browsed to.

These options can be changed at any time by opening the settings window by clicking the link in the top bar.  
