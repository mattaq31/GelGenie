package qupath.ext.gelgenie;

import javafx.beans.property.BooleanProperty;
import javafx.scene.control.MenuItem;
import javafx.stage.Stage;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.analysis.stats.Histogram;
import qupath.lib.common.GeneralTools;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.charts.Charts;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.plugins.CommandLinePluginRunner;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;
import qupath.opencv.tools.OpenCVTools;
import qupath.lib.gui.charts.HistogramPanelFX;
import qupath.lib.algorithms.IntensityFeaturesPlugin;
import qupath.lib.io.GsonTools;

import java.util.Map;



import javafx.scene.paint.Color;
import javafx.scene.layout.GridPane;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Random;
import java.util.ResourceBundle;

import static qupath.lib.gui.scripting.QPEx.getCurrentViewer;
import static qupath.lib.scripting.QP.*;

import qupath.ext.gelgenie.ui.ActivateUI;
import qupath.ext.gelgenie.graphics.GelGenieBarChart;


/**
 * This is a demo to provide a template for creating a new QuPath extension.
 * <p>
 * It doesn't do much - it just shows how to add a menu item and a preference.
 * See the code and comments below for more info.
 * <p>
 * <b>Important!</b> For your extension to work in QuPath, you need to make sure the name &amp; package
 * of this class is consistent with the file
 * <pre>
 *     /resources/META-INF/services/qupath.lib.gui.extensions.QuPathExtension
 * </pre>
 */
public class GelGenie implements QuPathExtension {

    private final static ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");
    private final static Logger logger = LoggerFactory.getLogger(QuPathExtension.class);
    private final static String EXTENSION_NAME = resources.getString("extension.title");
    private final static String EXTENSION_DESCRIPTION = resources.getString("extension.description");
    private final static Version EXTENSION_QUPATH_VERSION = Version.parse(resources.getString("extension.qupath.version"));


    /**
     * Flag whether the extension is already installed (might not be needed... but we'll do it anyway)
     */
    private boolean isInstalled = false;

    /**
     * A 'persistent preference' - showing how to create a property that is stored whenever QuPath is closed
     */
    private BooleanProperty enableExtensionProperty = PathPrefs.createPersistentPreference(
            "enableExtension", true);

    @Override
    public void installExtension(QuPathGUI qupath) {
        if (isInstalled) {
            logger.debug("{} is already installed", getName());
            return;
        }
        isInstalled = true;
        addPreference(qupath);
        addMenuItems(qupath);
    }

    /**
     * Demo showing how to add a persistent preference to the QuPath preferences pane.
     *
     * @param qupath
     */
    private void addPreference(QuPathGUI qupath) {
        qupath.getPreferencePane().addPropertyPreference(
                enableExtensionProperty,
                Boolean.class,
                "Enable my extension",
                EXTENSION_NAME,
                "Enable my extension");
    }

    /**
     * Demo showing how a new command can be added to a QuPath menu.
     *
     * @param qupath
     */
    private void addMenuItems(QuPathGUI qupath) {
        var menu = qupath.getMenu("Extensions>" + EXTENSION_NAME, true);
        MenuItem menuItem = new MenuItem("Show GUI");

        ActivateUI command = new ActivateUI(qupath);
        menuItem.setOnAction(e -> {
            command.run();
        });

        menuItem.disableProperty().bind(enableExtensionProperty.not());
        menu.getItems().add(menuItem);

        MenuItem menu2 = new MenuItem("My second menu item");
        menu2.setOnAction(e -> {
            ImageData<BufferedImage> imageData = getCurrentImageData();
            ImageServer<BufferedImage> server = imageData.getServer();

            Random random = new Random(8);

            double[] data2 = new double[100];
            for (int i = 0; i < 100; i++) {
                data2[i] = random.nextFloat()*100;
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
            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }

            Mat mat = OpenCVTools.imageToMat(img);
//			runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', imageData, "doMean": true);
            // Define output path (relative to project)
            var outputDir = buildFilePath(PROJECT_BASE_DIR, "Edited SegMaps");


			Histogram hist1 = new Histogram(data2, 15);

            GridPane pane = new GridPane();
            int row = 0;

            HistogramPanelFX histogramPanel = new HistogramPanelFX();


            var dhist = HistogramPanelFX.createHistogramData(hist1, false, Color.BLUE);

            histogramPanel.setShowTickLabels(false);
            histogramPanel.getChart().setAnimated(false);

            pane.add(histogramPanel.getChart(), 0, 0);

            histogramPanel.getHistogramData().setAll(dhist);

            try {
                Files.createDirectories(Path.of(outputDir));
            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
            var name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName());
            var path = buildFilePath(outputDir, name + " Edited SegMaps.tif");

            // Define how much to downsample during export (may be required for large images)
            double downsample = 1;

            // Create an ImageServer where the pixels are derived from annotations
//			def labelServer = new LabeledImageServer.Builder(imageData)
//					.backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
//					.downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
//					//.addLabel("Background", 2)      // Choose output labels (the order matters!)
//					.addLabel("Gel Band", 1)
//					.multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
//					.build();
//
//			// Write the image
//			writeImage(labelServer, path);
        });
        menu2.disableProperty().bind(enableExtensionProperty.not());
        menu.getItems().add(menu2);

        MenuItem menuArea = new MenuItem("Plot Band Areas");
        menuArea.setOnAction(e -> {
            ImageData<BufferedImage> imageData = getCurrentImageData();
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
            inten_plugin.runPlugin(new CommandLinePluginRunner<>(imageData), jsonargs);

            double[] all_areas = new double[annots.size()];
            double[] index_array = new double[annots.size()];

            int i = 0;

//            getAnnotationObjects().sort{it.getROI().getCentroidX()};
//          how would I assign names?  Ideally, it would start from top left and assign names top-down
            for (PathObject annot : annots){
                var mean_intensity = annot.getMeasurements().get("ROI: 1.00 px per pixel: Channel 1: Mean");
                var roi_area = annot.getROI().getArea();
                all_areas[i] = roi_area * mean_intensity; //perhaps can add this as a measurement to the annotation?
                annot.getMeasurementList().putMeasurement("IntensitySum", all_areas[i]);
                i++;
                index_array[i-1] = i;
            }

            GelGenieBarChart chart_var = new GelGenieBarChart();
            chart_var.plot(all_areas);

            Stage builder = Charts.scatterChart().
                    viewer(viewer).
                    title("My scatterplot").
                    series("Test", index_array, all_areas).
                    markerOpacity(0.5).
                    show();

        });
        menuArea.disableProperty().bind(enableExtensionProperty.not());
        menu.getItems().add(menuArea);

    }


    @Override
    public String getName() {
        return EXTENSION_NAME;
    }

    @Override
    public String getDescription() {
        return EXTENSION_DESCRIPTION;
    }

    @Override
    public Version getQuPathVersion() {
        return EXTENSION_QUPATH_VERSION;
    }

}
