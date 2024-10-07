/**
 * Copyright 2024 University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package qupath.ext.gelgenie.models;

import ai.djl.modality.cv.output.CategoryMask;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.process.ImageProcessor;

import javafx.application.Platform;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.opencv.core.CvType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.gelgenie.djl_processing.*;
import qupath.fx.dialogs.Dialogs;
import qupath.imagej.processing.RoiLabeling;
import qupath.imagej.tools.IJTools;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.regions.Padding;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.scripting.QP;
import qupath.opencv.dnn.DnnTools;
import qupath.opencv.dnn.OpenCVDnn;
import qupath.opencv.ops.ImageDataOp;
import qupath.opencv.ops.ImageOp;
import qupath.opencv.ops.ImageOps;
import qupath.opencv.tools.OpenCVTools;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.modality.cv.*;
import ai.djl.modality.cv.transform.*;
import ai.djl.repository.zoo.*;
import ai.djl.translate.*;
import ai.djl.training.util.*;
import ai.djl.modality.cv.ImageFactory;

import static qupath.lib.scripting.QP.addObjects;
import static qupath.opencv.tools.OpenCVTools.matToBufferedImage;


/**
 * Main class taking care of running segmentation models using OpenCV and onnx files.
 */
public class ModelRunner {
    private static final Logger logger = LoggerFactory.getLogger(ModelRunner.class);
    private static final double downsample = 1.0;
    private static boolean conn8 = true;

    private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");

    /**
     * Runs segmentation on a full image.
     *
     * @param model:     Model to be used for segmentation.
     * @param useDJL:    Boolean - set to true to use DJL, set to false to use OpenCV.
     * @param invertImage: Boolean - set to true to invert image before running model.
     * @param dataMaxNorm: Boolean - set to true to normalize by datatype max range instead of max pixel value in image
     * @param imageData: Image containing gel bands to be segmented     .
     * @return Collection of individual gel bands found in image
     * @throws IOException
     */
    public static Collection<PathObject> runFullImageInference(GelGenieModel model, boolean useDJL, boolean invertImage, boolean dataMaxNorm, ImageData<BufferedImage> imageData) throws IOException, TranslateException, ModelNotFoundException, MalformedModelException {

        ImageServer<BufferedImage> server = imageData.getServer();
        // Use the entire image at full resolution
        RegionRequest request = RegionRequest.createInstance(server, downsample);

        if(useDJL){
            return runDJLModel(model, imageData, request, model.getName().contains("nnUNet"), invertImage, dataMaxNorm);
        }
        else{
            if (model.getName().contains("nnUNet")) {
                Dialogs.showErrorMessage(resources.getString("ui.model-error.window-header"), resources.getString("error.model-issue"));
                return null;
            }
            return runOpenCVModel(model, imageData, request, invertImage, dataMaxNorm);
        }
    }

    /**
     * Runs segmentation on a full image, in a scripting-friendly format.
     * @param model: String-name of model to be used (must be available in model collection)
     * @param useDJL: Boolean - set to true to use DJL, set to false to use OpenCV.
     * @param invertImage: Boolean - set to true to invert image before running model.
     * @param dataMaxNorm: Boolean - set to true to normalize by datatype max range instead of max pixel value in image
     * @return Collection of individual gel bands found in image.
     * @throws IOException
     * @throws TranslateException
     * @throws ModelNotFoundException
     * @throws MalformedModelException
     */
    public static Collection<PathObject> runFullImageInference(String model, Boolean useDJL, Boolean invertImage, Boolean dataMaxNorm) throws IOException, TranslateException, ModelNotFoundException, MalformedModelException {
        GelGenieModel selectedModel = ModelInterfacing.loadModel(model);
        return runFullImageInference(selectedModel, useDJL, invertImage, dataMaxNorm, QP.getCurrentImageData());
    }

    /**
     * Runs segmentation on a full image, in a scripting-friendly format.  Additionally, automatically adds found annotations to QuPath's viewer.
     * @param model: String-name of model to be used (must be available in model collection)
     * @param useDJL: Boolean - set to true to use DJL, set to false to use OpenCV.
     * @param invertImage: Boolean - set to true to invert image before running model.
     * @param dataMaxNorm: Boolean - set to true to normalize by datatype max range instead of max pixel value in image
     * @return Collection of individual gel bands found in image.
     * @throws IOException
     * @throws TranslateException
     * @throws ModelNotFoundException
     * @throws MalformedModelException
     */
    public static Collection<PathObject> runFullImageInferenceAndAddAnnotations(String model, Boolean useDJL, Boolean invertImage, Boolean dataMaxNorm) throws TranslateException, ModelNotFoundException, MalformedModelException, IOException {
        Collection<PathObject> annotations = runFullImageInference(model, useDJL, invertImage, dataMaxNorm);
        for (PathObject annot : annotations) {
            annot.setPathClass(PathClass.fromString("Gel Band", 8000));
        }
        addObjects(annotations);

        return annotations;
    }

    /**
     * Runs model specifically within a selected annotation only.
     *
     * @param imageData:  Image containing selected annotation
     * @param annotation: Annotation within which to find bands
     * @return Collection of individual gel bands found in annotation area
     * @throws IOException
     */
    public static Collection<PathObject> runAnnotationInference(GelGenieModel model, boolean useDJL, boolean invertImage,
                                                                boolean dataMaxNorm,
                                                                ImageData<BufferedImage> imageData,
                                                                PathObject annotation) throws IOException, TranslateException, ModelNotFoundException, MalformedModelException {

        ImageServer<BufferedImage> server = imageData.getServer();

        // Slice out the selected annotation at full resolution
        RegionRequest request = RegionRequest.createInstance(server.getPath(), 1.0, annotation.getROI());
        if(useDJL){
            return runDJLModel(model, imageData, request, model.getName().contains("nnUNet"), invertImage, dataMaxNorm);
        }
        else{
            if (model.getName().contains("nnUNet")) {
                Dialogs.showErrorMessage(resources.getString("ui.model-error.window-header"), resources.getString("error.model-issue"));
                return null;
            }
            return runOpenCVModel(model, imageData, request, invertImage, dataMaxNorm);
        }
    }

    /**
     * Runs ONNX model using openCV.  Formats output mask into QuPath annotations.  Image has to be split into tiles
     * since OpenCV models (post version 4.6) do not support dynamic input sizes.
     * @param imageData:   Actual full image
     * @param request:     Region from which pixels to be extracted
     * @return Collection of annotations containing segmented gel bands
     * @throws IOException
     */
    private static Collection<PathObject> runOpenCVModel(GelGenieModel model, ImageData<BufferedImage> imageData, RegionRequest request, Boolean invertImage, Boolean dataMaxNorm) throws IOException {

        try (var scope = new PointerScope()) { // pointer scope allows for automatic memory management

            int padValue = 64;
            int[] fullPadding = {padValue, padValue, padValue, padValue}; // padding selected to ensure that when image is tiled, minimal artifacts will appear due to arbitrary breaks in the image at tile boundaries

            OpenCVDnn dnnModel = DnnTools.builder(model.getOnnxFile().toString()).size( 1024, 1024).build(); // creates opencv model from ONNX file - model size is fixed due to opencv issue with dynamic inputs

            BufferedImage image = imageData.getServer().readRegion(request);

            // Create required intermediate Mat image
//            Mat predImage = new Mat();
            Mat convertedImage;

            // for 8-bit images, normalizing by dividing by 255 works in all cases
            if (image.getType() == BufferedImage.TYPE_BYTE_GRAY) {
                convertedImage = OpenCVTools.imageToMat(image);
                convertedImage.convertTo(convertedImage, CvType.CV_32F);
                opencv_core.dividePut(convertedImage, 255.0);
            }
            // however, for other cases, normalizing by the max bit value could be problematic since it may be
            // extremely high and will completely squash small values.  In this case, the user is given the choice
            // between normalizing with the max bit value (default) or max pixel value in the image.
            else{
                convertedImage = convertAndNormalizeNonStandardImage(image, imageData, dataMaxNorm);
            }

            // inverts image if requested
            if (invertImage) {
                opencv_core.subtract(Mat.ones(convertedImage.size(), convertedImage.type()).asMat(), convertedImage, convertedImage);
            }

            Mat paddedImage = new Mat();

            // applies zero padding here
            opencv_core.copyMakeBorder(convertedImage, paddedImage, fullPadding[0], fullPadding[1], fullPadding[2], fullPadding[3], opencv_core.BORDER_CONSTANT, new Scalar(0.0));

            // runs model - model has to be a fixed size due to OpenCV limitations (post 4.6)
            ImageOp dataOp = ImageOps.ML.dnn(dnnModel, 1024, 1024, Padding.symmetric(padValue));

            Mat predImage = dataOp.apply(paddedImage); // the image padding will be removed automatically by the tiling system

            // The below processes the two channels from the openCV model into a single segmentation map

            // first, the output is split into two different channels
            List<Mat> splitchannels = OpenCVTools.splitChannels(predImage);

            // indexers and a new output image are created
            Mat segmentedMat = new Mat(request.getHeight(), request.getWidth(), CvType.CV_32F);
            Indexer backgroundIndexer = splitchannels.get(0).createIndexer();
            Indexer foregroundIndexer = splitchannels.get(1).createIndexer();
            Indexer newIndexer = segmentedMat.createIndexer();

            // the split channels are compared - the output pixel is a positive if the foreground channel is larger than the background channel
            for (int i = 0; i < request.getHeight(); i++) {
                for (int j = 0; j < request.getWidth(); j++) {
                    int classSelection = 0;
                    if (foregroundIndexer.getDouble(i, j) >= backgroundIndexer.getDouble(i, j)) {
                        classSelection = 1;
                    }
                    ((FloatIndexer) newIndexer).put(i, j, classSelection);
                }
            }

            // final labelling occurs on the segmented image
            return findSplitROIs(segmentedMat, request);
        }
    }

    /**
     * Runs the selected model using DJL, which involves generating a pre-processing pipeline, using the model for
     * inference and then splitting output pixels into annotations.
     * IMPORTANT: DJL models only run on 8-bit images - 16-bit images are automatically converted to 8-bit ones.
     * This should not affect final gel quantitation, however.
     * @param imageData: The actual image pixel data.
     * @param request: The request corresponding to the image pixels.
     * @return A list of gel annotations found by the model.
     * @throws IOException
     * @throws ModelNotFoundException
     * @throws MalformedModelException
     * @throws TranslateException
     */
    private static Collection<PathObject> runDJLModel(GelGenieModel model, ImageData<BufferedImage> imageData, RegionRequest request, Boolean nnunetConfig, Boolean invertImage, Boolean dataMaxNorm) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {

        Device device = PytorchManager.getDevice();
        int imageWidth = request.getWidth();
        int imageHeight = request.getHeight();

        // TODO: this should probably now be removed since DJL loading has been moved entirely to the DJL extension.  Should still check for PyTorch though.
        checkPyTorchLibrary();

        Translator<Image, CategoryMask> translator;

        // prepares pipeline for preprocessing and running models here
        if (nnunetConfig){ // nnunet do not require any padding (this is done internally in the torchscript model)
            if (invertImage){
                translator = NnUNetSegmentationTranslator.builder()
                        .addTransform(createToTensorTransform(device))
                        .addTransform(new ChannelSquisher())
                        .addTransform(new ImageInvert())
                        .build(imageWidth, imageHeight);
            }
            else {
                translator = NnUNetSegmentationTranslator.builder()
                        .addTransform(createToTensorTransform(device))
                        .addTransform(new ChannelSquisher())
                        .build(imageWidth, imageHeight);
            }
        }
        else {

            int[] fullPadding = getPadding(imageHeight, imageWidth, 32);

            if (invertImage){  // inverts image (only if requested)
                translator = GelSegmentationTranslator.builder()
                        .addTransform(createToTensorTransform(device))
                        .addTransform(new DivisibleSizePad(fullPadding[0], fullPadding[1], fullPadding[2], fullPadding[3]))
                        .addTransform(new ChannelSquisher())
                        .addTransform(new ImageInvert())
                        .build(request.getWidth(), request.getHeight(), fullPadding[2], fullPadding[3], fullPadding[0], fullPadding[1]);
            }
            else {
                translator = GelSegmentationTranslator.builder()
                        .addTransform(createToTensorTransform(device))
                        .addTransform(new DivisibleSizePad(fullPadding[0], fullPadding[1], fullPadding[2], fullPadding[3]))
                        .addTransform(new ChannelSquisher())
                        .build(request.getWidth(), request.getHeight(), fullPadding[2], fullPadding[3], fullPadding[0], fullPadding[1]);
            }
        }

        ImageFactory factory = ImageFactory.getInstance();

        BufferedImage img = imageData.getServer().readRegion(request);

        // DJL's Image class is flawed - it only works properly with 8-bit images.
        // To make it work with other image types and match openCV's output, need to convert to 8-bit explicitly
        // However, since the image needs to be converted to 8-bit, some resolution is lost, which results in
        // some minor mismatches with the OpenCV version.
        // TODO: can this be resolved entirely by using Mat instead of Image for the DJL engine?  Need to investigate.
        Image image;
        if (img.getType() == BufferedImage.TYPE_BYTE_GRAY) {
            image = factory.fromImage(img);
        }
        else{
            // as with opencv, user is given the choice between max pixel and max range normalization for non 8-bit images
            Mat convertedImage = convertAndNormalizeNonStandardImage(img, imageData, dataMaxNorm);

            // converts to 8-bit image
            opencv_core.multiplyPut(convertedImage, 255.0);
            convertedImage.convertTo(convertedImage, CvType.CV_8U);

            // extracts DJL's Image from the 8-bit image
            image = factory.fromImage(matToBufferedImage(convertedImage));
        }

        Criteria<Image, CategoryMask> criteria = Criteria.builder()
                .setTypes(Image.class, CategoryMask.class)
                .optModelPath(model.getTSFile().toPath())
                .optDevice(device)
                .optOption("mapLocation", "true") // this model requires mapLocation for GPU
                .optEngine("PyTorch")
                .optTranslator(translator)
                .optProgress(new ProgressBar()).build();

        int[][] maskOrig; // predicts masks from image, making sure to autoclose the model and predictor when complete
        try(ZooModel<Image, CategoryMask> modelConstruct = criteria.loadModel()) {
            try (Predictor<Image, CategoryMask> predictor = modelConstruct.newPredictor()) {
                CategoryMask mask = predictor.predict(image);
                maskOrig = mask.getMask(); // extracts out integer array for downstream processing
            }
        }

        try (var scope = new PointerScope()) { // pointer scope allows for automatic memory management
            // The below converts the integer array into openCV Mat.  Unsure if there is a more elegant way to do this.
            Mat imMat = new Mat(request.getHeight(), request.getWidth(), CvType.CV_32F);
            Indexer indexer = imMat.createIndexer();
            for (int i = 0; i < request.getHeight(); i++) {
                for (int j = 0; j < request.getWidth(); j++) {
                    ((FloatIndexer) indexer).put(i, j, maskOrig[i][j]);
                }
            }

            return findSplitROIs(imMat, request); // final splitter operation
        }
    }

    private static Mat convertAndNormalizeNonStandardImage(BufferedImage img, ImageData<BufferedImage> imageData, boolean dataMaxNorm){

        // extracts image with OpenCV and converts to float
        Mat normImage = OpenCVTools.imageToMat(img);
        normImage.convertTo(normImage, CvType.CV_32F);

        // if an RGB image provided, average channels into a single channel
        if (normImage.channels() > 1) {
            var temp = normImage.reshape(1, normImage.rows() * normImage.cols());
            opencv_core.reduce(temp, temp, 1, opencv_core.REDUCE_AVG);
            normImage = temp.reshape(1, normImage.rows());
        }

        if (!dataMaxNorm){
            // normalises by the maximum pixel value in the image
            double maxVal = Arrays.stream(OpenCVTools.extractDoubles(normImage)).max().getAsDouble();
            opencv_core.dividePut(normImage, maxVal);
        }
        else{
            opencv_core.dividePut(normImage, imageData.getServer().getPixelType().getUpperBound().doubleValue());
        }

        return normImage;
    }

    public static Transform createToTensorTransform(Device device) {
        logger.debug("Creating ToTensor transform");
        if (PytorchManager.isMPS(device))
            return new MpsSupport.ToTensor32();
        else
            return new ToTensor();
    }

    /**
     * Attempts to check if PyTorch library is available and if not, downloads it. TODO: improve legibility of this area
     */
    public static void checkPyTorchLibrary() {
        try {
            // Ensure PyTorch engine is available
            if (!PytorchManager.hasPyTorchEngine()) {
                Platform.runLater(() -> Dialogs.showInfoNotification(resources.getString("ui.pytorch.window-header"), resources.getString("ui.pytorch-downloading")));
                PytorchManager.getEngineOnline();
            }
            // Ensure model is available - any prompts allowing the user to cancel
            // should have been displayed already
        }  catch (Exception e) {
            Platform.runLater(() -> Dialogs.showErrorMessage(resources.getString("ui.pytorch.window-header"), e.getLocalizedMessage()));
        }
    }

    /**
     * Takes a mask found by both types of models and attempts to split out annotations according to
     * whether pixels touch or not.
     * @param maskMat OpenCV Mask found by segmentation models
     * @param request Underlying image pixels corresponding to mask
     * @return List of ROIs that have been split out
     */
    public static Collection<PathObject> findSplitROIs(Mat maskMat, RegionRequest request){

        // Convert to an ImageJ-friendly form
        ImagePlus imp = OpenCVTools.matToImagePlus("Result", maskMat);

        // combines adjacent pixels into individual labels
        ImageProcessor ipLabels = RoiLabeling.labelImage(imp.getProcessor(), 0.5F, conn8);

        // converts labels into qupath annotations
        Roi[] roisIJ = RoiLabeling.labelsToConnectedROIs(ipLabels, (int) Math.ceil(ipLabels.getStatistics().max));

        Collection<PathObject> convertedROIs = Arrays.stream(roisIJ).map(roiIJ -> {
            // adjusted origin for ROI conversion so it re-fits the original image (otherwise would be set to 0)
            ROI roi = IJTools.convertToROI(roiIJ, -request.getMinX(), -request.getMinY(), downsample, request.getImagePlane());
            return PathObjects.createAnnotationObject(roi);
        }).collect(Collectors.toList());

        return convertedROIs;
    }

    /**
     * Gets the padding required to make the image divisible by 32.  This is a direct copy of the function used in python.
     * @param imageHeight: Original image height
     * @param imageWidth: Original image width
     * @param multiplier: Multiplier (in this case always 32)
     * @return: Array of padding values to be applied to image (left, right, top, bottom)
     */
    private static int[] getPadding(int imageHeight, int imageWidth, int multiplier){
        // 32-divisible padding occurs evenly around an image, identical to the way it is done in python
        int newVert = multiplier * (imageHeight / multiplier + 1);
        int newHoriz = multiplier * (imageWidth / multiplier + 1);

        int topPad = (newVert - imageHeight) / 2;
        int bottomPad = newVert - imageHeight - topPad;
        int leftPad = (newHoriz - imageWidth) / 2;
        int rightPad = newHoriz - imageWidth - leftPad;

        return new int[]{leftPad, rightPad, topPad, bottomPad};
    }
}
