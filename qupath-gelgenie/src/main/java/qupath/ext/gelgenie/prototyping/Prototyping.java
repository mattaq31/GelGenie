package qupath.ext.gelgenie.prototyping;

import javafx.scene.layout.GridPane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import org.bytedeco.opencv.opencv_core.Mat;
import qupath.ext.gelgenie.graphics.GelGenieBarChart;
import qupath.ext.gelgenie.tools.ImageTools;
import qupath.lib.algorithms.IntensityFeaturesPlugin;
import qupath.lib.analysis.stats.Histogram;
import qupath.lib.common.GeneralTools;
import qupath.lib.gui.charts.Charts;
//import qupath.lib.gui.charts.HistogramPanelFX;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.io.GsonTools;
import qupath.lib.objects.PathObject;
//import qupath.lib.plugins.CommandLinePluginRunner;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;
import qupath.opencv.tools.OpenCVTools;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;
import java.util.Random;

import static qupath.lib.gui.scripting.QPEx.getCurrentViewer;
import static qupath.lib.scripting.QP.*;


/**
Collection of functions used while building main extension.  Should be deleted before public exporting.
 TODO: review and delete when done.
 */
public class Prototyping {

    public static void testingPixels() {
        ImageData<BufferedImage> imageData = getCurrentImageData();
        ImageServer<BufferedImage> server = imageData.getServer();

        Random random = new Random(8);

        double[] data2 = new double[100];
        for (int i = 0; i < 100; i++) {
            data2[i] = random.nextFloat() * 100;
        }
        QuPathViewer viewer = getCurrentViewer();
        Collection<PathObject> cells = getCellObjects();
        Collection<PathObject> annots = getAnnotationObjects();
        Stage builder = Charts.scatterChart().
                viewer(viewer).
                title("My scatterplot").
                series("Test", data2, data2).
                markerOpacity(0.5).
                show();

        ROI roi = getSelectedROI();

        RegionRequest request = RegionRequest.createInstance(server.getPath(), 1.0, roi);
        BufferedImage img;
        try {
            img = server.readRegion(request);
        } catch (
                IOException ex) {
            throw new RuntimeException(ex);
        }

        Mat mat = OpenCVTools.imageToMat(img);
        //			runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', imageData, "doMean": true);
        // Define output path (relative to project)
        var outputDir = buildFilePath(PROJECT_BASE_DIR, "Edited SegMaps");


        Histogram hist1 = new Histogram(data2, 15);

        GridPane pane = new GridPane();
        int row = 0;

//        HistogramPanelFX histogramPanel = new HistogramPanelFX();
//
//
//        var dhist = HistogramPanelFX.createHistogramData(hist1, false, Color.BLUE);
//
//        histogramPanel.setShowTickLabels(false);
//        histogramPanel.getChart().setAnimated(false);
//
//        pane.add(histogramPanel.getChart(), 0, 0);
//
//        histogramPanel.getHistogramData().setAll(dhist);
//
//        try {
//            Files.createDirectories(Path.of(outputDir));
//        } catch (IOException ex) {
//            throw new RuntimeException(ex);
//        }
//        var name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName());
//        var path = buildFilePath(outputDir, name + " Edited SegMaps.tif");

    }

    public static void testingGraph(){ // CURRENTLY BROKEN AS PLUGIN RUNNER DOES NOT WORK
        ImageData<BufferedImage> imageData = getCurrentImageData();
        ImageServer<BufferedImage> server = imageData.getServer();
        QuPathViewer viewer = getCurrentViewer();
        Collection<PathObject> annots = getAnnotationObjects();

        selectAnnotations();

        IntensityFeaturesPlugin inten_plugin = new IntensityFeaturesPlugin();

        Map<String, ?> intensityargs = Map.of(
                "region", "ROI",
                "downsample", 1.0,
                "channel1", true,
                "doMean", true
        );
        var jsonargs = GsonTools.getInstance().toJson(intensityargs);
//        var cl_plugin = new CommandLinePluginRunner(imageData);
//        inten_plugin.runPlugin(cl_plugin, jsonargs);

        double[] all_areas = new double[annots.size()];
        double[] new_areas = new double[annots.size()];
        double[] index_array = new double[annots.size()];

        int i = 0;

//            getAnnotationObjects().sort{it.getROI().getCentroidX()};
//          how would I assign names?  Ideally, it would start from top left and assign names top-down
        for (PathObject annot : annots){
            var mean_intensity = annot.getMeasurements().get("ROI: 1.00 px per pixel: Channel 1: Mean");
            var roi_area = annot.getROI().getArea();
            all_areas[i] = roi_area * mean_intensity; //perhaps can add this as a measurement to the annotation?

            double[] all_pixels = ImageTools.extractAnnotationPixels(annot, server);
            double pixel_sum = Arrays.stream(all_pixels).sum(); // TODO: this seems to spit out integers only not doubles
            GelGenieBarChart chart_var = new GelGenieBarChart();
            chart_var.plot(all_pixels, 40);

            annot.getMeasurementList().putMeasurement("IntensitySum", all_areas[i]);
            annot.getMeasurementList().putMeasurement("IntensitySum2", pixel_sum);

            i++;
            index_array[i-1] = i;
            break;
        }

//            GelGenieBarChart chart_var = new GelGenieBarChart();
//            chart_var.plot(all_areas, 10);

//            Stage builder = Charts.scatterChart().
//                    viewer(viewer).
//                    title("My scatterplot").
//                    series("Test", index_array, all_areas).
//                    markerOpacity(0.5).
//                    show();
    }
}
