GelGenie QuPath Extension
==============================
The below is meant as a quick-reference guide for all the main features of the GelGenie QuPath extension.  If you are a developer, please browse the source code [here](src/main/java/qupath/ext/gelgenie) for further technical details, or jump to the development mode installation [instructions](#installing-codebase-in-development-mode).
## Installation
Follow the instructions on the main page [here](https://github.com/mattaq31/GelGenie/tree/main?tab=readme-ov-file#installing-the-qupath-gelgenie-extension).

## Loading Images or Creating Projects

The extension can be used in two modes - either by directly loading a temporary image or creating a project.  A project gives you the advantage of having your segmentation results permanently saved in a directory of your choosing.
- For direct image loading, simply drag-and-drop your image into QuPath, in the same way the extension was first installed.  If a prompt is shown asking you to set the image type, feel free to change the setting to `Always auto-estimate type (don't prompt)` (shown below).  GelGenie does not use the image type for any of its operations.  Images loaded in this way are temporary and any segmentations made will be lost when QuPath is closed (unless you choose to save data somewhere).
<p align="center">
<img src="./screenshots/image_estimation_prompt.png" alt="Silencing image estimation prompt" width="300">
</p>

- To use a project, simply create an empty folder anywhere on your file system, then select the `Create Project` option from the GelGenie interface.  Select the empty folder you just created the deposit your project there.  Any images you now add to the project (drag-and-drop as before) will now be recorded in this project.  This also includes any segmentations you generate.  Please note that the project will **not** save a copy of the underlying image data but simply store a reference to its location.  Deleting the original image data will also cause it to be lost in QuPath (QuPath will prompt you to search for its new location if you've just moved the data and not deleted it).  Best practice is to store your images in a separate (ideally permanent) location.
<p align="center">
<img src="./screenshots/project_creation.png" alt="Creating a project" width="500">
</p>

-  With a project, you can switch between images through the menu on the left, save data at any time and also script operations on all images in your project (more info [here](#scripting)).
<p align="center">
<img src="./screenshots/switch_between_project_images_here.png" alt="Switching between project images" width="400">
</p>

## Opening the Graphical Interface
GelGenie can be opened directly from the QuPath interface by clicking on the Extensions menu and navigating to the `Activate GelGenie` option.
<p align="center">
<img src="./screenshots/open_extension.png" alt="How to activate GelGenie" width="600">
</p>

## Downloading and Running Segmentation Models
To run your first segmentation, follow these steps:
- Load in an image (either directly or through a project).
- Open the GelGenie interface.
- Select a model and download it to your local PC by clicking on the download icon.
  - **The 'Universal Model' is recommended for general-purpose use.**  If results are unsatisfactory, especially with very sharp bands, **the 'Sharp Band Model' is a good alternative.**
  - You can get more info on each model by clicking on the info button next to the download icon.
  - You are free to play around with all the other models available, but most are prototype models that do not offer increased performance in many cases.
  - All `nnUNet` models can only be run with DJL (see below) and are very slow without a GPU.
<p align="center">
<img src="./screenshots/model_inference.png" alt="Downloading and running models" width="400">
</p>

- After clicking the download button, the extension will show a prompt that the model has started downloading.  Once complete, another prompt will be shown.  Please be patient - slow internet connections can take a while to download the models, although in most cases the download should be done in seconds.
<p align="center">
<img src="./screenshots/download_notif_1.png" alt="Download notification 1" width="300">
<img src="./screenshots/download_notif_2.png" alt="Download notification 2" width="300">
</p>

- After the download is complete, the `Identify Bands` button will be enabled. Click on this button to initialize the segmentation process.  There are two paths that can be taken for segmentation, which are explained in the next sections.
- Four checkboxes are available above the `Identify Bands` button.  These can affect the segmentation process as follows:
  - **`Find bands in entire image`** - Selecting this will run segmentation on the entire image (default).
  - **`Find bands in selected region`** - Selecting this will run segmentation only within the selected annotation (more details on annotations [here](#editing-band-segmentation)).
  - **`Delete previous bands`** - Selecting this will delete all previous segmentation results before generating new ones.
  - **`Light bands on dark background`** - Selecting this will assume that bands are lighter than the background (and vice-versa).  The extension will attempt to auto-assign the value of this checkbox but can sometimes make mistakes.  Make sure to fix this setting if it is incorrect for the current image.
- Once a model is downloaded, it is always available for use and an internet connection is no longer required.
### Direct CPU Inference (OpenCV Mode)
- The easiest way to run models is in direct CPU inference mode using OpenCV, which is the default setting.  
- This mode is fast, requires no preparation and will work well with all systems and models (except `nnUNet` models).
- After clicking on the `Identify Bands` button, GelGenie will show a quick notification that inference has started.  Typically, the process should take just a few seconds on most normal gel images (expect a longer delay for very large .tif images).
- When complete, the segmented bands will be shown directly on the image, as shown below.  Segmentation results can be adjusted and quantified following the guides in the next sections.
<p align="center">
<img src="./screenshots/example_seg.png" alt="Example segmentation result" width="600">
</p>

### Deep Java Library (DJL) GPU Inference
- Using DJL requires more setup but allows you to make use of GPUs available on your system to accelerate inference.  Apple Silicon GPUs are also supported (using MPS).
- Before using DJL, you will need to install QuPath's DJL Extension.  This can be done by following the instructions [here](https://github.com/qupath/qupath-extension-djl) (drag-and-drop the jar file in the same way as GelGenie).
- Next, open the DJL extension from the usual extension menu, and select the `Manage DJL engines' option.
<p align="center">
<img src="./screenshots/djl_setup.png" alt="How to open DJL extension" width="600">
</p>
- From the resulting menu, click on the `Check/Download` button to install the PyTorch engine.  Ensure the engine is downloaded properly before moving on.  This step only needs to be done once, after which the engine is permanently available.
<p align="center">
<img src="./screenshots/djl_download_1.png" alt="Downloading PyTorch engine" width="300">
<img src="./screenshots/djl_download_2.png" alt="PyTorch engine download successful" width="300">
</p>

- After the engine is installed, you are now ready to run models using DJL.  To configure GelGenie to use DJL, switch to the `Advanced` tab and check the `Run Models using DJL` checkbox.  To use an available GPU, select it from the dropdown list underneath the DJL checkbox.
- Switching back to the `Band Search` tab will now allow you to use DJL to run inference with the usual `Identify Bands' button.  Keep in mind the following:
  - The first time the GPU is used, there will be a short delay.  Subsequent runs will be much faster.  On an Apple Silicon GPU, typical images are segmented almost instantaneously.
  - You should only run `nnUNet` models in GPU mode, as CPU mode will take a very long time (minutes).
- On restarting GelGenie, make sure to re-enable the DJL checkbox if you wish to use it again.  
<p align="center">
<img src="./screenshots/djl_gelgenie_setup.png" alt="Configuring DJL in GelGenie" width="300">
</p>

## Editing Band Segmentation

### Adjusting Borders

### Changing Display and Labels

## Band Quantification
### Direct Raw Measurement

### Background Correction

### Results Display

### Exporting Results

## Scripting

## Updating or Deleting the Extension

The extension version you currently have installed can be verified from the Extensions -> manage extensions tool.

<p align="center">
<img src="./screenshots/open_extension_manager.png" alt="How to open extension manager" width="600">
</p>

To update the extension, simply click on the update button to automatically download the latest release and have it installed.  Make sure to restart QuPath after the update.

To delete the extension, the same extension manager also has a delete button you can use to clear out the extension from your QuPath installation.  To completely delete all models downloaded by GelGenie, you should also click on the `Open extension directory` button and delete the `gelgenie` folder.

<p align="center">
<img src="./screenshots/extension_manager.png" alt="Extension manager info" width="600">
</p>

## Further QuPath Info
For further tips, tricks and more info on what is achievable within QuPath (alongside the GelGenie extension), please consult the main documentation from [here](https://qupath.github.io).
## Installing Codebase in Development Mode
If interested in adding new features to the extension, the best way to have direct access to the source code and all of QuPath's features is to use [IntelliJ](https://www.jetbrains.com/idea/) as your main IDE.  To start developing, follow the steps outlined [here](https://github.com/qupath/qupath-extension-template#set-up-in-an-ide-optional) to A) download QuPath's source code and B) setup everything in IntelliJ.  Keep in mind the following details:
- The GelGenie extension needs to be in the folder **beside** that of the main QuPath source code to be properly recognized.  To achieve this, first clone the GelGenie repository, and then clone the QuPath repository within the GelGenie folder.  A gitignore statement has already been added to prevent the QuPath repository from conflicting with the GelGenie repository.
- When the above is complete, you can proceed to add the project to IntelliJ as described in the link above.
- Finally, don't forget to add the statement `includeFlat qupath-gelgenie` to the `qupath/settings.gradle` file in the QuPath repository to enable the extension.
- If everything has been setup correctly, you should be able to build QuPath from scratch, with the extension included within it.  You can also use IntelliJ's debug feature to investigate issues and help with development.  
- Keep in mind that QuPath is constantly evolving, and the latest commit on GitHub might contain features not available in the current release.

