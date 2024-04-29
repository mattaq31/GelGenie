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

package qupath.ext.gelgenie.tools;

import org.bytedeco.opencv.global.opencv_core;
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
     * @param invertImage: Set to true to invert image before extracting pixels
     */
    public static double[] extractAnnotationPixels(PathObject annotation, ImageServer<BufferedImage> server, boolean invertImage) {

        // creates a mask of the correct shape within the rectangular region bordering an annotation
        BufferedImage im_mask = BufferedImageTools.createROIMask((int) Math.ceil(annotation.getROI().getBoundsWidth()),
                (int) Math.ceil(annotation.getROI().getBoundsHeight()), annotation.getROI(),
                annotation.getROI().getBoundsX(), annotation.getROI().getBoundsY(), 1.0);

        // creates a request for the full rectangular region
        RegionRequest request = RegionRequest.createInstance(server.getPath(), 1.0, annotation.getROI());
        BufferedImage img;
        try {
            img = server.readRegion(request);
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }

        // selects pixels which are in the mask and not the outer edges of the rectangular region
        return extractPixelsfromMask(OpenCVTools.imageToMat(img), OpenCVTools.imageToMat(im_mask),
                false, invertImage, server.getPixelType().getUpperBound().doubleValue());
    }
    /**
     * Extracts pixels from a selected annotation, with an OpenCV image provided instead of an ImageServer.
     *
     * @param annotation: Specific annotation containing data of interest
     * @param fullImage: OpenCV Mat containing full image of interest
     * @param invertImage: Set to true to invert image before extracting pixels
     * @param invertMax:  Need to pre-calculate max value for inversion as the value is more difficult to extract from a Mat
     */
    public static double[] extractAnnotationPixelsFromMat(PathObject annotation, Mat fullImage, Boolean invertImage, double invertMax) {

        // creates a mask of the correct shape within the image
        BufferedImage im_mask = BufferedImageTools.createROIMask((int) Math.ceil(annotation.getROI().getBoundsWidth()),
                (int) Math.ceil(annotation.getROI().getBoundsHeight()), annotation.getROI(),
                annotation.getROI().getBoundsX(), annotation.getROI().getBoundsY(), 1.0);

        // crops out rectangular region bordering annotation from opencv image
        Mat croppedImage = OpenCVTools.crop(fullImage,
                (int) annotation.getROI().getBoundsX(),
                (int) annotation.getROI().getBoundsY(),
                (int) annotation.getROI().getBoundsWidth(),
                (int) annotation.getROI().getBoundsHeight());

        // extracts masked pixels as normal
        return extractPixelsfromMask(croppedImage, OpenCVTools.imageToMat(im_mask), false,
                invertImage, invertMax);
    }

    /**
     * Extracts the pixels bordering a specified annotation.
     * @param annotation: Annotation around which to extract pixels
     * @param server: Server corresponding to open image
     * @param pixelBorder: Width of border around annotation
     * @param invertImage: Set to true to invert image before extracting pixels
     * @return Array of pixels from border surrounding annotation
     */
    public static double[] extractLocalBackgroundPixels(PathObject annotation, ImageServer<BufferedImage> server, int pixelBorder,
                                                        boolean invertImage) {

        // creates rectangular region bordering annotation
        ImageRegion extendedLocalRegion = createAnnotationImageFrame(annotation, pixelBorder);
        RegionRequest request = RegionRequest.createInstance(server.getPath(), 1.0, extendedLocalRegion);
        BufferedImage img;
        try {
            img = server.readRegion(request);
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }

        // gets actual annotation shape mask
        BufferedImage im_mask = BufferedImageTools.createROIMask(request.getWidth(),
                request.getHeight(), annotation.getROI(), request.getX(), request.getY(), 1.0);

        return extractPixelsfromMask(OpenCVTools.imageToMat(img), OpenCVTools.imageToMat(im_mask), true, invertImage,
                server.getPixelType().getUpperBound().doubleValue());
    }

    /**
     * Given a rectangular image and a mask situated within the image, extract the pixels that are present
     * (or not present) within the mask.
     * @param rectangular_mat: Rectangular frame of image
     * @param mat_mask: Mask containing irregular shape within rectangular frame
     * @param ExtractNotMaskedPixels: Set to true to extract pixels outside of mask rather than within
     * @param invertImage: Set to true to invert image before extracting pixels
     * @param invertMax: The max pixel value for the particular image in use (required to be able to invert)
     * @return Array of pixels that are present within the mask
     */
    public static double[] extractPixelsfromMask(Mat rectangular_mat, Mat mat_mask, Boolean ExtractNotMaskedPixels, Boolean invertImage, double invertMax) {

        double[] mask_pixels = OpenCVTools.extractDoubles(mat_mask);
        double[] main_pixels;
        if (invertImage) {
            Mat subtractor = Mat.ones(rectangular_mat.size(), rectangular_mat.type()).asMat();
            opencv_core.multiplyPut(subtractor, invertMax);
            opencv_core.subtract(subtractor, rectangular_mat, rectangular_mat);
        }
        main_pixels = OpenCVTools.extractDoubles(rectangular_mat);

        ArrayList<Double> final_pixels = new ArrayList<Double>();

        double targetValue = 255.0;
        if(ExtractNotMaskedPixels){
            targetValue = 0.0;
        }

        // extracts pixels matching the mask
        for (int j = 0; j < mask_pixels.length; j++) {
            if (mask_pixels[j] == targetValue) {
                final_pixels.add(main_pixels[j]);
            }
        }

        return final_pixels.stream().mapToDouble(d -> d).toArray();
    }

    /**
     * Creates a region bordering an annotation, with the specified pixel width.
     * @param annotation: Annotation to extract border around
     * @param pixelBorder: Width of pixel border (on each side)
     * @return
     */
    public static ImageRegion createAnnotationImageFrame(PathObject annotation, int pixelBorder){

        int x1 = (int) annotation.getROI().getBoundsX() - pixelBorder;
        int y1 = (int) annotation.getROI().getBoundsY() - pixelBorder;
        int x2 = (int) Math.ceil(annotation.getROI().getBoundsX() + annotation.getROI().getBoundsWidth()) + pixelBorder;
        int y2 = (int) Math.ceil(annotation.getROI().getBoundsY() + annotation.getROI().getBoundsHeight()) + pixelBorder;

        return ImageRegion.createInstance(x1, y1, x2 - x1, y2 - y1, annotation.getROI().getZ(), annotation.getROI().getT());
    }

}
