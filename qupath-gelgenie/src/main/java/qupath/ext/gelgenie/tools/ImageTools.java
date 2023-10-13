package qupath.ext.gelgenie.tools;

import org.bytedeco.opencv.opencv_core.Mat;
import qupath.lib.awt.common.BufferedImageTools;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.regions.ImageRegion;
import qupath.lib.regions.RegionRequest;
import qupath.opencv.tools.OpenCVTools;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;

/**
 * This class contains useful functions for dealing with and extracting image pixels
 */
public class ImageTools {

    /**
     * Extracts pixels from a selected annotation, even when annotation does not have a regular shape.
     *
     * @param annotation: Specific annotation containing data of interest
     * @param server:     Object containing image data pixels
     */
    public static double[] extractAnnotationPixels(PathObject annotation, ImageServer<BufferedImage> server) {

        BufferedImage im_mask = BufferedImageTools.createROIMask((int) Math.ceil(annotation.getROI().getBoundsWidth()),
                (int) Math.ceil(annotation.getROI().getBoundsHeight()), annotation.getROI(),
                annotation.getROI().getBoundsX(), annotation.getROI().getBoundsY(), 1.0);

        RegionRequest request = RegionRequest.createInstance(server.getPath(), 1.0, annotation.getROI());
        BufferedImage img;
        try {
            img = server.readRegion(request);
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }

        return extractPixelsfromMask(img, im_mask, false);
    }

    public static double[] extractLocalBackgroundPixels(PathObject annotation, ImageServer<BufferedImage> server, int pixelBorder) {

        ImageRegion extendedLocalRegion = createAnnotationImageFrame(annotation, pixelBorder);
        RegionRequest request = RegionRequest.createInstance(server.getPath(), 1.0, extendedLocalRegion);

        BufferedImage im_mask = BufferedImageTools.createROIMask(request.getWidth(),
                request.getHeight(), annotation.getROI(), request.getX(), request.getY(), 1.0);

        BufferedImage img;
        try {
            img = server.readRegion(request);
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }

        return extractPixelsfromMask(img, im_mask, true);
    }

    /**
     * Given a rectangular image and a mask situated within the image, extract the pixels that are present
     * (or not present) within the mask.
     * @param image: Rectangular frame of image
     * @param mask: Mask containing irregular shape within rectangular frame
     * @param ExtractNotMaskedPixels: Set to true to extract pixels outside of mask rather than within.
     * @return Array of pixels that are present within the mask
     */
    public static double[] extractPixelsfromMask(BufferedImage image, BufferedImage mask, Boolean ExtractNotMaskedPixels) {
        Mat rectangular_mat = OpenCVTools.imageToMat(image);
        Mat mat_mask = OpenCVTools.imageToMat(mask);
        double[] mask_pixels = OpenCVTools.extractDoubles(mat_mask);
        double[] main_pixels = OpenCVTools.extractDoubles(rectangular_mat);
        ArrayList<Double> final_pixels = new ArrayList<Double>();

        double targetValue = 255.0;
        if(ExtractNotMaskedPixels){
            targetValue = 0.0;
        }

        for (int j = 0; j < main_pixels.length; j++) {
            if (mask_pixels[j] == targetValue) {
                final_pixels.add(main_pixels[j]);
            }
        }
        return final_pixels.stream().mapToDouble(d -> d).toArray();
    }

    public static ImageRegion createAnnotationImageFrame(PathObject annotation, int pixelBorder){

        int x1 = (int) annotation.getROI().getBoundsX() - pixelBorder;
        int y1 = (int) annotation.getROI().getBoundsY() - pixelBorder;
        int x2 = (int) Math.ceil(annotation.getROI().getBoundsX() + annotation.getROI().getBoundsWidth()) + pixelBorder;
        int y2 = (int) Math.ceil(annotation.getROI().getBoundsY() + annotation.getROI().getBoundsHeight()) + pixelBorder;

        return ImageRegion.createInstance(x1, y1, x2 - x1, y2 - y1, annotation.getROI().getZ(), annotation.getROI().getT());
    }

}
