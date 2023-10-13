package qupath.ext.gelgenie.tools;

import ij.ImagePlus;
import ij.gui.Roi;
import ij.plugin.filter.MaximumFinder;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;

import java.io.File;

import org.bytedeco.opencv.opencv_core.Mat;
import qupath.imagej.processing.RoiLabeling;
import qupath.imagej.tools.IJTools;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.regions.Padding;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.GeometryROI;
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
import java.util.Arrays;
import java.util.Collection;
import java.util.Objects;
import java.util.stream.Collectors;

import static qupath.lib.scripting.QP.*;

public class openCVModelRunner {

    private final double downsample = 1.0;
    private String modelPath;
    private Padding paddingMode = Padding.empty();

    private double threshold = 0.5;
    private double tolerance = 0.1;
    private boolean excludeOnEdges = true;
    private boolean isEDM = false;
    private boolean conn8 = true;
    private MaximumFinder maxFinder = new MaximumFinder();

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

    public Collection<PathObject> runFullImageInference(ImageData<BufferedImage> imageData) throws IOException {

        ImageServer<BufferedImage> server = imageData.getServer();
        // Use the entire image at full resolution
        RegionRequest request = RegionRequest.createInstance(server, downsample);

        // Avoid tiling by using the image dimensions
        // Here, specify tiles to help ensure that our troubles don't come from weird dimensions
        int inputWidth = (int) (Math.ceil(server.getWidth() / 32) * 32);
        int inputHeight = (int) (Math.ceil(server.getHeight() / 32) * 32);

        return runModel(inputWidth, inputHeight, imageData, request);
    }

    public Collection<PathObject> runAnnotationInference(ImageData<BufferedImage> imageData, PathObject annotation) throws IOException {

        ImageServer<BufferedImage> server = imageData.getServer();

        // Slice out the selected annotation at full resolution
        RegionRequest request = RegionRequest.createInstance(server.getPath(), 1.0, annotation.getROI());

        // Avoid tiling by using the image dimensions
        // Here, specify tiles to help ensure that our troubles don't come from weird dimensions
        int inputWidth = (int) (Math.ceil(request.getWidth() / 32) * 32);
        int inputHeight = (int) (Math.ceil(request.getHeight() / 32) * 32);

        return runModel(inputWidth, inputHeight, imageData, request);
    }

    private Collection<PathObject> runModel(int inputWidth, int inputHeight, ImageData<BufferedImage> imageData, RegionRequest request) throws IOException {

        OpenCVDnn dnnModel = DnnTools.builder(modelPath).size(inputWidth, inputHeight).build();

        ImageDataOp dataOp = ImageOps.buildImageDataOp().appendOps(
                ImageOps.Normalize.percentile(0.1, 99.9),
                ImageOps.Channels.mean(),
                ImageOps.ML.dnn(dnnModel, inputWidth, inputHeight, paddingMode),
                ImageOps.Normalize.channelSoftmax(1)
        );

        Mat result = dataOp.apply(imageData, request);
        // Convert to an ImageJ-friendly form for now
        ImagePlus imp = OpenCVTools.matToImagePlus("Result", result);

        // Apply the maximum finder to the second channel
        ImageProcessor ip = imp.getStack().getProcessor(2);

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
