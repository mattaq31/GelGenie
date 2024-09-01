GelGenie QuPath Extension
==============================
The below is meant as a quick-reference guide for all the main features of the GelGenie QuPath extension.  If you are a developer, please browse the source code [here](src/main/java/qupath/ext/gelgenie) for further technical details, or jump to the development mode installation [instructions](#installing-codebase-in-development-mode).
## Installation
Follow the instructions on the main page [here](https://github.com/mattaq31/GelGenie/tree/main?tab=readme-ov-file#installing-the-qupath-gelgenie-extension).
## Opening the Interface
GelGenie 
## Loading Images or Creating Projects

## Downloading and Running Models

### Direct CPU Inference (OpenCV Mode)

### Deep Java Library (DJL) GPU Inference

## Editing Band Segmentation

### Adjusting Borders

### Changing Display and Labels

## Band Quantification
### Direct Raw Measurement

### Background Correction

### Results Display

### Exporting Results

## Scripting

## Updating the Extension

## Further QuPath Info

## Installing Codebase in Development Mode
If interested in adding new features to the extension, the best way to have direct access to the source code and all of QuPath's features is to use [IntelliJ](https://www.jetbrains.com/idea/) as your main IDE.  To start developing, follow the steps outlined [here](https://github.com/qupath/qupath-extension-template#set-up-in-an-ide-optional) to A) download QuPath's source code and B) setup everything in IntelliJ.  Keep in mind the following details:
- The GelGenie extension needs to be in the folder **beside** that of the main QuPath source code to be properly recognized.  To achieve this, first clone the GelGenie repository, and then clone the QuPath repository within the GelGenie folder.  A gitignore statement has already been added to prevent the QuPath repository from conflicting with the GelGenie repository.
- When the above is complete, you can proceed to add the project to IntelliJ as described in the link above.
- Finally, don't forget to add the statement `includeFlat qupath-gelgenie` to the `qupath/settings.gradle` file in the QuPath repository to enable the extension.
- If everything has been setup correctly, you should be able to build QuPath from scratch, with the extension included within it.  You can also use IntelliJ's debug feature to investigate issues and help with development.  
- Keep in mind that QuPath is constantly evolving, and the latest commit on GitHub might contain features not available in the current release.

