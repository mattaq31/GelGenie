package qupath.ext.gelgenie.tools;

import org.bytedeco.opencv.opencv_core.Mat;
import qupath.lib.awt.common.BufferedImageTools;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.regions.RegionRequest;
import qupath.opencv.tools.OpenCVTools;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;

/**
This class contains useful functions for dealing with and extracting image pixels
*/
public class ImageTools {

    /**
     * Extracts pixels from a selected annotation, even when annotation does not have a regular shape.
     @param annotation: Specific annotation containing data of interest
     @param server: Object containing image data pixels
     */
    public static double[] extractAnnotationPixels(PathObject annotation, ImageServer<BufferedImage> server){

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
        Mat rectangular_mat = OpenCVTools.imageToMat(img);
        Mat mat_mask = OpenCVTools.imageToMat(im_mask);

        double[] mask_pixels = OpenCVTools.extractDoubles(mat_mask);
        double[] main_pixels = OpenCVTools.extractDoubles(rectangular_mat);
        ArrayList<Double> final_pixels = new ArrayList<Double>();

        for (int j = 0; j < main_pixels.length; j++){
            if(mask_pixels[j] == 255.0){
                final_pixels.add(main_pixels[j]);
            }
        }
        return final_pixels.stream().mapToDouble(d -> d).toArray();
    }

}
