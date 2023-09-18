package qupath.ext.gelgenie.ui;

import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.beans.binding.BooleanBinding;
import javafx.beans.binding.ObjectBinding;
import javafx.beans.binding.StringBinding;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.value.ObservableBooleanValue;
import javafx.beans.value.ObservableValue;
import javafx.concurrent.Task;
import javafx.concurrent.Worker;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.ContentDisplay;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleButton;
import javafx.stage.Stage;
import javafx.util.StringConverter;
import org.controlsfx.control.action.Action;
import org.controlsfx.control.action.ActionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
//import qupath.ext.wsinfer.ProgressListener;
//import qupath.ext.wsinfer.WSInfer;
//import qupath.ext.wsinfer.models.WSInferModel;
//import qupath.ext.wsinfer.models.WSInferModelCollection;
//import qupath.ext.wsinfer.models.WSInferUtils;
import qupath.ext.gelgenie.graphics.EmbeddedBarChart;
import qupath.ext.gelgenie.graphics.GelGenieBarChart;
import qupath.ext.gelgenie.tools.ImageTools;
import qupath.lib.common.ThreadTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.commands.Commands;
//import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathTileObject;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.objects.hierarchy.events.PathObjectSelectionListener;
import qupath.lib.plugins.workflow.DefaultScriptableWorkflowStep;
import qupath.lib.roi.interfaces.ROI;

import java.awt.image.BufferedImage;
import java.util.Collection;
import java.util.Objects;
import java.util.ResourceBundle;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;

import static qupath.lib.scripting.QP.*;

/**
This is the main class that controls things happening in the UI window.  Most functionality starts from a function here.
 A few functions were cannibalised from the wsinfer extension.  TODO: These need to be removed/adapted before final release.
 */
public class UIController {

    private static final Logger logger = LoggerFactory.getLogger(UIController.class);

    public QuPathGUI qupath;
    private ObjectProperty<ImageData<BufferedImage>> imageDataProperty = new SimpleObjectProperty<>();
//    private MessageTextHelper messageTextHelper;

    @FXML
    private Label labelMessage;
    //    @FXML
//    private ChoiceBox<WSInferModel> modelChoiceBox;
    @FXML
    private Button runButton;
    @FXML
    private Button downloadButton;
    @FXML
    private ChoiceBox<String> deviceChoices;
    @FXML
    private ToggleButton toggleSelectAllAnnotations;
    @FXML
    private ToggleButton toggleSelectAllDetections;
    @FXML
    private ToggleButton toggleDetectionFill;
    @FXML
    private ToggleButton toggleDetections;
    @FXML
    private ToggleButton toggleAnnotations;
    @FXML
    private Slider sliderOpacity;
    @FXML
    private Spinner<Integer> spinnerNumWorkers;
    @FXML
    private TextField tfModelDirectory;

    @FXML
    private BarChart<String, Number> bandChart;

    private final static ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");

    private Stage measurementMapsStage;

    private final ExecutorService pool = Executors.newSingleThreadExecutor(ThreadTools.createThreadFactory("gelgenie", true));

    private final ObjectProperty<Task<?>> pendingTask = new SimpleObjectProperty<>();

    @FXML
    private void initialize() {
        logger.info("Initializing...");

        this.qupath = QuPathGUI.getInstance();
        this.imageDataProperty.bind(qupath.imageDataProperty());
        configureDisplayToggleButtons();
        configureRunInferenceButton();
        bandChart.setBarGap(0);
        bandChart.setCategoryGap(0);
        bandChart.setLegendVisible(false);
        bandChart.getXAxis().setLabel("Pixel Intensity");
        bandChart.getYAxis().setLabel("Frequency");
    }

    private void configureActionToggleButton(Action action, ToggleButton button) {
        ActionUtils.configureButton(action, button);
        button.setContentDisplay(ContentDisplay.GRAPHIC_ONLY);
    }

    private void configureDisplayToggleButtons() {
        var actions = qupath.getOverlayActions();
        configureActionToggleButton(actions.FILL_DETECTIONS, toggleDetectionFill);
        configureActionToggleButton(actions.SHOW_DETECTIONS, toggleDetections);
        configureActionToggleButton(actions.SHOW_ANNOTATIONS, toggleAnnotations);
    }

    private void configureRunInferenceButton() {
        // Disable the run button while a task is pending, or we have no model selected, or download is required
        runButton.disableProperty().bind(
                imageDataProperty.isNull()
                        .or(pendingTask.isNotNull())
        );
    }

    public void runInference() {
        ImageData<BufferedImage> imageData = getCurrentImageData();
        PathObject annot = getSelectedObject();
        ImageServer<BufferedImage> server = imageData.getServer();
        double[] all_pixels = ImageTools.extractAnnotationPixels(annot, server); // extracts a list of pixels matching the specific selected annotation
//        GelGenieBarChart chart_var = new GelGenieBarChart(); - this is an explicit barchart (not needed)
//        chart_var.plot(all_pixels, 40);
        EmbeddedBarChart outbar = new EmbeddedBarChart();
        bandChart.getData().clear(); // removes previous data
        bandChart.getData().addAll(outbar.plot(all_pixels, 40)); // adds new data TODO: x-axis ticks are broken on first run - how to fix?
        // TODO: change functionality to happen on selection of a new band.  What happens if multiple bands selected?  Should average them all and show an indicator of how many bands are averaged....
    }

    @FXML
    private void openMeasurementMaps(ActionEvent event) {
        // Try to use existing action, to avoid creating a new stage
        // TODO: Replace this if QuPath v0.5.0 provides direct access to the action
        //       since that should be more robust (and also cope with language changes)
        var action = qupath.lookupActionByText("Show measurement maps");
        if (action != null) {
            action.handle(event);
            return;
        }
        // Fallback in case we couldn't get the action
        if (measurementMapsStage == null) {
            logger.warn("Creating a new measurement map stage");
            measurementMapsStage = Commands.createMeasurementMapDialog(QuPathGUI.getInstance());
        }
        measurementMapsStage.show();
    }

    @FXML
    private void openDetectionTable() {
        Commands.showAnnotationMeasurementTable(qupath, imageDataProperty.get());
    }


}
