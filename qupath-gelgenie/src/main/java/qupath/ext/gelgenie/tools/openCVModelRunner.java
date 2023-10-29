package qupath.ext.gelgenie.tools;

import ai.djl.modality.cv.output.CategoryMask;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.plugin.filter.MaximumFinder;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;

import java.io.File;

import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.opencv.core.CvType;
import qupath.imagej.processing.RoiLabeling;
import qupath.imagej.tools.IJTools;
import qupath.lib.gui.prefs.PathPrefs;
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
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
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
public class openCVModelRunner {
    // TODO: many of these settings could be adjustable or user-adjustable
    private final double downsample = 1.0;
    private String modelPath;
    private Padding paddingMode = Padding.empty();

    private double threshold = 0.5;
    private double tolerance = 0.1;
    private boolean excludeOnEdges = true;
    private boolean isEDM = false;
    private boolean conn8 = true;
    private MaximumFinder maxFinder = new MaximumFinder();

    private final boolean useDJL = true;

    /**
     * Main constructor takes care of finding model and downloading if necessary.
     *
     * @param modelName: Specified model to load and run with.
     */
    public openCVModelRunner(String modelName) {
        if (Objects.equals(modelName, "Prototype-UNet-July-29-2023")) {

            String userPath = String.valueOf(PathPrefs.getDefaultQuPathUserDirectory());
            String modelFolder = Paths.get(userPath, "gelgenie", "prototyping", "Prototype-UNet-July-29-2023").toString();
            modelPath = modelFolder + File.separator + "exported_checkpoint.onnx";
            File folder = new File(modelFolder);
            File file = new File(modelPath);

            if (!file.exists()) {
                try {
                    Files.createDirectories(folder.toPath());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                URL url = null;
                try {
                    url = new URL("https://huggingface.co/mattaq/Prototype-UNet-Gel-Band-Finder-July-29-2023/resolve/main/onnx_format/exported_checkpoint.onnx");
                } catch (MalformedURLException e) {
                    throw new RuntimeException(e);
                }
                try {
                    ModelInterfacing.downloadURLToFile(url, file);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }

        } else {
            throw new IllegalArgumentException("Other models not yet available."); // TODO: update here when these are available
        }
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
            int inputWidth = (int) (Math.ceil(server.getWidth() / 32) * 32);
            int inputHeight = (int) (Math.ceil(server.getHeight() / 32) * 32);
            return runOpenCVModel(inputWidth, inputHeight, imageData, request);
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
            int inputWidth = (int) (Math.ceil(server.getWidth() / 32) * 32);
            int inputHeight = (int) (Math.ceil(server.getHeight() / 32) * 32);
            return runOpenCVModel(inputWidth, inputHeight, imageData, request);
        }
    }

    /**
     * Main function that runs model and formats output.
     *
     * @param inputWidth:  Image width
     * @param inputHeight: Image height
     * @param imageData:   Actual full image
     * @param request:     Region from which pixels to be extracted
     * @return Collection of annotations containing segmented gel bands
     * @throws IOException TODO: investigate all parameters here and see if anything can be optimised
     */
    private Collection<PathObject> runOpenCVModel(int inputWidth, int inputHeight, ImageData<BufferedImage> imageData, RegionRequest request) throws IOException {

        OpenCVDnn dnnModel = DnnTools.builder(modelPath).size(inputWidth, inputHeight).build();

        // Inference pipeline, which reduces images to a single channel and normalises
        ImageDataOp dataOp = ImageOps.buildImageDataOp().appendOps(
                ImageOps.Normalize.percentile(0.1, 99.9),
                ImageOps.Channels.mean(),
                ImageOps.ML.dnn(dnnModel, inputWidth, inputHeight, paddingMode),
                ImageOps.Normalize.channelSoftmax(1)
        );

        Mat result = dataOp.apply(imageData, request);

        return findSplitROIs(result, request);
    }

    private Collection<PathObject>  runDJLModel(ImageData<BufferedImage> imageData, RegionRequest request) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        // TODO: ToTensor() assumes data is between 0 and 255, what to do about 16-bit images?
        // TODO: Need to add prompt for user to download pytorch library
        // TODO: go through parameters and see if anything needs to be updated
        // TODO: full documentation
        Translator<Image, CategoryMask> translator = gelSegmentationTranslator.builder()
                .addTransform(new ToTensor())
                .addTransform(new divisibleSizePad(32))
                .addTransform(new channelSquisher())
                .build(request.getWidth(), request.getHeight());

        ImageFactory factory = ImageFactory.getInstance();

        BufferedImage img = imageData.getServer().readRegion(request);

        Image image = factory.fromImage(img);
        Criteria<Image, CategoryMask> criteria = Criteria.builder()
                .setTypes(Image.class, CategoryMask.class)
                .optModelPath(Paths.get("/Users/matt/Desktop/torchscript_model.pt"))// todo: remove harcoding
                .optOption("mapLocation", "true") // this model requires mapLocation for GPU
                .optEngine("PyTorch")
                .optTranslator(translator)
                .optProgress(new ProgressBar()).build();

        ZooModel model = criteria.loadModel();

        Predictor<Image, CategoryMask> predictor = model.newPredictor();

        CategoryMask mask = predictor.predict(image);
        int[][] maskOrig = mask.getMask();

        Mat imMat = new Mat(request.getHeight(), request.getWidth(), CvType.CV_32F);

        Indexer indexer = imMat.createIndexer();

        for (int i=0; i<request.getHeight(); i++){
            for (int j=0; j<request.getWidth(); j++) {
                ((FloatIndexer) indexer).put(i, j, maskOrig[i][j]);
            }
        }
        return findSplitROIs(imMat, request);
    }

    private Collection<PathObject> findSplitROIs(Mat maskMat, RegionRequest request){

        // Convert to an ImageJ-friendly form for now
        ImagePlus imp = OpenCVTools.matToImagePlus("Result", maskMat);
        // Apply the maximum finder to the second channel
        ImageProcessor ip = imp.getStack().getProcessor(1);
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
