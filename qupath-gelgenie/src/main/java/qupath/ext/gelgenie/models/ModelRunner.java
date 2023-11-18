package qupath.ext.gelgenie.models;

import ai.djl.modality.cv.output.CategoryMask;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.plugin.filter.MaximumFinder;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;

import javafx.application.Platform;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.opencv.core.CvType;
import qupath.ext.gelgenie.tools.ChannelSquisher;
import qupath.ext.gelgenie.tools.DivisibleSizePad;
import qupath.ext.gelgenie.tools.GelSegmentationTranslator;
import qupath.imagej.processing.RoiLabeling;
import qupath.imagej.tools.IJTools;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.regions.Padding;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;
import qupath.opencv.dnn.DnnTools;
import qupath.opencv.dnn.OpenCVDnn;
import qupath.opencv.ops.ImageDataOp;
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

/**
 * Main class taking care of running segmentation models using OpenCV and onnx files.
 */
public class ModelRunner {
    // TODO: many of these settings could be adjustable or user-adjustable
    private final double downsample = 1.0;
    private GelGenieModel baseModel;
    private Padding paddingMode = Padding.empty();
    private double threshold = 0.5;
    private double tolerance = 0.1;
    private boolean excludeOnEdges = true;
    private boolean isEDM = false;
    private boolean conn8 = true;
    private MaximumFinder maxFinder = new MaximumFinder();

    private final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");

    private final boolean useDJL;

    /**
     * Main constructor takes care of finding model and downloading if necessary.
     *
     * @param model: Specified model to load and run with (must be in GelGenie format).
     */
    public ModelRunner(GelGenieModel model, boolean useDJL) {
        baseModel = model;
        this.useDJL = useDJL;
    }

    /**
     * Runs segmentation on a full image.
     *
     * @param imageData: Image containing gel bands to be segmented
     * @return Collection of individual gel bands found in image
     * @throws IOException
     */
    public Collection<PathObject> runFullImageInference(ImageData<BufferedImage> imageData) throws IOException, TranslateException, ModelNotFoundException, MalformedModelException {

        ImageServer<BufferedImage> server = imageData.getServer();
        // Use the entire image at full resolution
        RegionRequest request = RegionRequest.createInstance(server, downsample);

        if(useDJL){
            return runDJLModel(imageData, request);
        }
        else{
            return runOpenCVModel(imageData, request);
        }
    }

    /**
     * Runs model specifically within a selected annotation only.
     *
     * @param imageData:  Image containing selected annotation
     * @param annotation: Annotation within which to find bands
     * @return Collection of individual gel bands found in annotation area
     * @throws IOException
     */
    public Collection<PathObject> runAnnotationInference(ImageData<BufferedImage> imageData, PathObject annotation) throws IOException, TranslateException, ModelNotFoundException, MalformedModelException {

        ImageServer<BufferedImage> server = imageData.getServer();

        // Slice out the selected annotation at full resolution
        RegionRequest request = RegionRequest.createInstance(server.getPath(), 1.0, annotation.getROI());
        if(useDJL){
            return runDJLModel(imageData, request);
        }
        else{
            return runOpenCVModel(imageData, request);
        }
    }

    /**
     * Runs ONNX model using openCV.  Formats output mask into QuPath annotations.
     * @param imageData:   Actual full image
     * @param request:     Region from which pixels to be extracted
     * @return Collection of annotations containing segmented gel bands
     * @throws IOException TODO: investigate all parameters here and see if anything can be optimised
     */
    private Collection<PathObject> runOpenCVModel(ImageData<BufferedImage> imageData, RegionRequest request) throws IOException {

        int inputWidth = (int) (Math.ceil((double) request.getWidth() / 32) * 32);
        int inputHeight = (int) (Math.ceil((double) request.getHeight() / 32) * 32);

        OpenCVDnn dnnModel = DnnTools.builder(baseModel.getOnnxFile().toString()).size(inputWidth, inputHeight).build();

        // Inference pipeline, which reduces images to a single channel and normalises
        ImageDataOp dataOp = ImageOps.buildImageDataOp().appendOps(
                ImageOps.Normalize.percentile(0.1, 99.9),
                ImageOps.Channels.mean(),
                ImageOps.ML.dnn(dnnModel, inputWidth, inputHeight, paddingMode),
                ImageOps.Normalize.channelSoftmax(1)
        );

        Mat result = dataOp.apply(imageData, request);

        return findSplitROIs(result, request, 2); // final splitter operation
    }

    /**
     * Runs the selected model using DJL, which involves generating a pre-processing pipeline, using the model for
     * inference and then splitting output pixels into annotations.
     * @param imageData: The actual image pixel data.
     * @param request: The request corresponding to the image pixels.
     * @return A list of gel annotations found by the model.
     * @throws IOException
     * @throws ModelNotFoundException
     * @throws MalformedModelException
     * @throws TranslateException
     */
    private Collection<PathObject> runDJLModel(ImageData<BufferedImage> imageData, RegionRequest request) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        // TODO: ToTensor() assumes data is between 0 and 255, what to do about 16-bit images?
        // TODO: go through parameters and see if anything needs to be updated
        // TODO: full documentation
        // TOD: move image height/width finding in the input processing function, so that model doesn't need to be generated from scratch each time

        checkPyTorchLibrary();
        Translator<Image, CategoryMask> translator = GelSegmentationTranslator.builder()
                .addTransform(new ToTensor())
                .addTransform(new DivisibleSizePad(32))
                .addTransform(new ChannelSquisher())
                .build(request.getWidth(), request.getHeight());

        ImageFactory factory = ImageFactory.getInstance();

        BufferedImage img = imageData.getServer().readRegion(request);

        Image image = factory.fromImage(img);
        Criteria<Image, CategoryMask> criteria = Criteria.builder()
                .setTypes(Image.class, CategoryMask.class)
                .optModelPath(baseModel.getTSFile().toPath())
                .optOption("mapLocation", "true") // this model requires mapLocation for GPU
                .optEngine("PyTorch")
                .optTranslator(translator)
                .optProgress(new ProgressBar()).build();

        ZooModel model = criteria.loadModel();

        Predictor<Image, CategoryMask> predictor = model.newPredictor();

        CategoryMask mask = predictor.predict(image);
        int[][] maskOrig = mask.getMask(); // extracts out integer array for downstream processing

        // The below converts the integer array into openCV Mat.  Unsure if there is a more elegant way to do this.
        Mat imMat = new Mat(request.getHeight(), request.getWidth(), CvType.CV_32F);

        Indexer indexer = imMat.createIndexer();

        for (int i=0; i<request.getHeight(); i++){
            for (int j=0; j<request.getWidth(); j++) {
                ((FloatIndexer) indexer).put(i, j, maskOrig[i][j]);
            }
        }
        return findSplitROIs(imMat, request, 1); // final splitter operation
    }

    /**
     * Attempts to check if PyTorch library is available and if not, downloads it. TODO: improve legibility of this area
     */
    private void checkPyTorchLibrary(){
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
    private Collection<PathObject> findSplitROIs(Mat maskMat, RegionRequest request, int maximumFinderChannel){

        // Convert to an ImageJ-friendly form for now
        ImagePlus imp = OpenCVTools.matToImagePlus("Result", maskMat);
        // Apply the maximum finder to the second channel
        ImageProcessor ip = imp.getStack().getProcessor(maximumFinderChannel);
        ByteProcessor bpDetected = maxFinder.findMaxima(ip, tolerance, threshold, ij.plugin.filter.MaximumFinder.SEGMENTED, excludeOnEdges, isEDM);
        ImageProcessor ipLabels = RoiLabeling.labelImage(bpDetected, 0.5F, conn8);
        Roi[] roisIJ = RoiLabeling.labelsToConnectedROIs(ipLabels, (int) Math.ceil(ipLabels.getStatistics().max));

        Collection<PathObject> convertedROIs = Arrays.stream(roisIJ).map(roiIJ -> {
            // adjusted origin for ROI conversion so it re-fits the original image (otherwise would be set to 0)
            ROI roi = IJTools.convertToROI(roiIJ, -request.getMinX(), -request.getMinY(), downsample, request.getImagePlane());
            return PathObjects.createAnnotationObject(roi);
        }).collect(Collectors.toList());
        return convertedROIs;
    }
}
