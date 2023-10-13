package qupath.ext.gelgenie.ui;

import javafx.beans.binding.Bindings;
import javafx.beans.property.*;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.beans.binding.ObjectBinding;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.concurrent.Task;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.stage.Stage;
import org.controlsfx.control.action.Action;
import org.controlsfx.control.action.ActionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.gelgenie.graphics.EmbeddedBarChart;
import qupath.ext.gelgenie.tools.ImageTools;
import qupath.ext.gelgenie.tools.openCVModelRunner;
import qupath.lib.common.ThreadTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.commands.Commands;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.events.PathObjectSelectionListener;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static qupath.lib.scripting.QP.*;

/**
 * This is the main class that controls things happening in the UI window.  Most functionality starts from a function here.
 * A few functions were cannibalised from the wsinfer extension.  TODO: These need to be removed/adapted before final release.
 */
public class UIController {
    private static final Logger logger = LoggerFactory.getLogger(UIController.class);

    public QuPathGUI qupath;
    private ObjectProperty<ImageData<BufferedImage>> imageDataProperty = new SimpleObjectProperty<>();
    //    private ObjectProperty<PathObject> selObjectProperty = new SimpleObjectProperty<>();
    @FXML
    private Label labelMessage;
    @FXML
    private Button runButton;
    @FXML
    private Button tableButton;
    @FXML
    private Button bandButton;
    @FXML
    private Button downloadButton;
    @FXML
    private ToggleButton toggleBandNames;
    @FXML
    private ToggleButton toggleAnnotations;

    @FXML
    private CheckBox runFullImage;
    @FXML
    private CheckBox runSelected;
    @FXML
    private CheckBox deletePreviousBands;

    @FXML
    private CheckBox enableGlobalBackground;
    @FXML
    private CheckBox enableLocalBackground;
    @FXML
    private Spinner<Integer> localSensitivity;

    @FXML
    private Button globalBackgroundSelector;

    @FXML
    private BarChart<String, Number> bandChart;

    @FXML
    private ChoiceBox<String> modelChoiceBox;

    private final static ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");

    private Stage measurementMapsStage;

    private final ExecutorService pool = Executors.newSingleThreadExecutor(ThreadTools.createThreadFactory("gelgenie", true));

    private final ObjectProperty<Task<?>> pendingTask = new SimpleObjectProperty<>();

    private SelectedObjectCounter selectedObjectCounter;

    @FXML
    private void initialize() {
        logger.info("Initializing GelGenie GUI");

        this.qupath = QuPathGUI.getInstance(); // linking to QuPath Instance
        this.imageDataProperty.bind(qupath.imageDataProperty());
        this.selectedObjectCounter = new SelectedObjectCounter(this.imageDataProperty);  // main listener that tracks selected annotations and changes in selection

        ChangeListener<Boolean> changeListener = (observable, oldValue, newValue) -> {

            if (selectedObjectCounter.numSelectedAnnotations.get() != 0) { // triggers an update of the histogram display whenever a different band is selected
                // this internal testing condition is crucial to prevent crashing when creating new annotations
                Collection<PathObject> actionableAnnotations = new ArrayList<>();
                for (PathObject annot : getSelectedObjects()) {
                    if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Gel Band")) {
                        actionableAnnotations.add(annot);
                    }
                }

                if (!actionableAnnotations.isEmpty()) {
                    BandHistoDisplay(actionableAnnotations);
                } else {
                    bandChart.getData().clear();
                }

            } else {
                bandChart.getData().clear();
            }
        };
        selectedObjectCounter.annotationSelected.addListener(changeListener);

        configureDisplayToggleButtons(); // styles QuPath linked buttons
        configureButtonInteractivity(); // sets rules for visibility of certain buttons
        configureCheckBoxes(); // sets rules for checkboxes

        modelChoiceBox.getItems().add("Prototype-UNet-July-29-2023");
        modelChoiceBox.setValue("Prototype-UNet-July-29-2023");

        // setting properties for single band update chart
        bandChart.setBarGap(0);
        bandChart.setCategoryGap(0);
        bandChart.setLegendVisible(false);
        bandChart.getXAxis().setLabel("Pixel Intensity");
        bandChart.getYAxis().setLabel("Frequency");

        logger.info("GelGenie GUI loaded without errors");
    }

    /**
     * Links the model checkboxes together so that when one is selected, the other must be off
     */
    private void configureCheckBoxes() {
        runFullImage.selectedProperty().addListener(new ChangeListener<Boolean>() {
            @Override
            public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) {
                runSelected.setSelected(!newValue);
            }
        });

        runSelected.selectedProperty().addListener(new ChangeListener<Boolean>() {
            @Override
            public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) {
                runFullImage.setSelected(!newValue);
            }
        });
    }

    /**
     * Links buttons with their respective graphics from the general QuPath interface.
     */
    private void configureActionToggleButton(Action action, ToggleButton button) {
        ActionUtils.configureButton(action, button);
        button.setContentDisplay(ContentDisplay.GRAPHIC_ONLY);
    }

    /**
     * Links buttons with their respective graphics from the general QuPath interface.  There are only 2 for GelGenie.
     */
    private void configureDisplayToggleButtons() {
        var actions = qupath.getOverlayActions();
        configureActionToggleButton(actions.SHOW_NAMES, toggleBandNames);
        configureActionToggleButton(actions.SHOW_ANNOTATIONS, toggleAnnotations);
    }

    private void configureButtonInteractivity() {
        // Disables the run button while a task is pending, or we have no model selected, or download is required
        runButton.disableProperty().bind(imageDataProperty.isNull().or(pendingTask.isNotNull()));
        tableButton.disableProperty().bind(imageDataProperty.isNull().or(pendingTask.isNotNull()));
        globalBackgroundSelector.disableProperty().bind(this.selectedObjectCounter.numSelectedAnnotations.isEqualTo(0));
    }

    public void BandHistoDisplay(Collection<PathObject> annotations) {

        ImageData<BufferedImage> imageData = getCurrentImageData();
        bandChart.getData().clear(); // removes previous data

        Collection<double[]> dataList = new ArrayList<>();

        for (PathObject annot : annotations) {
            ImageServer<BufferedImage> server = imageData.getServer();
            double[] all_pixels = ImageTools.extractAnnotationPixels(annot, server); // extracts a list of pixels matching the specific selected annotation
            dataList.add(all_pixels);
        }

        EmbeddedBarChart outbar = new EmbeddedBarChart();
        ObservableList<XYChart.Series<String, Number>> allPlots = outbar.plot(dataList, 40);
        bandChart.getData().addAll(allPlots); // adds new data TODO: x-axis ticks are broken on first run - how to fix?

    }

    /**
     * Runs the segmentation model on the provided image or selected annotation, generating annotations for each located band.
     *
     * @throws IOException
     */
    public void runBandInference() throws IOException {
        ImageData<BufferedImage> imageData = getCurrentImageData();// todo: need to handle situation where no image available and prompt user?
        openCVModelRunner modelRunner = new openCVModelRunner("Prototype-UNet-July-29-2023");

        Collection<PathObject> newBands = null;
        if (runFullImage.isSelected()) { // runs model on entire image
            newBands = modelRunner.runFullImageInference(imageData);
        } else if (runSelected.isSelected()) { // runs model on data within selected annotation only
            newBands = modelRunner.runAnnotationInference(imageData, getSelectedObject());
        }

        if (deletePreviousBands.isSelected()) { //removes all annotations before adding new ones
            for (PathObject annot : getAnnotationObjects()) {
                if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Gel Band")) {
                    removeObject(annot, false);
                }
            }
        }

        assert newBands != null; // todo: is this enough?
        for (PathObject annot : newBands) {
            annot.setPathClass(PathClass.fromString("Gel Band", 8000));
        }

        addObjects(newBands);
    }

    public void populateTable() {
        TableRootCommand tableCommand = new TableRootCommand(qupath, "gelgenie_table",
                "Data Table", true, enableGlobalBackground.isSelected(),
                enableLocalBackground.isSelected(), localSensitivity.getValue()); // activation class from ui folder
        tableCommand.run();
    }

    @FXML
    private void setGlobalBackgroundPatch() {
        PathObject annot = getSelectedObject();
        PathClass gbClass = PathClass.fromString("Global Background", 80);
        annot.setPathClass(gbClass);
    }


    /**
     * Helper class for maintaining a count of selected annotations and detections,
     * determined from an ImageData property (whose value may change).
     * This addresses the awkwardness of attaching/detaching listeners.  Adjusted from the WsInfer QuPath extension.
     */
    private static class SelectedObjectCounter {

        private ObjectProperty<ImageData<?>> imageDataProperty = new SimpleObjectProperty<>();

        private PathObjectSelectionListener selectionListener = this::selectedPathObjectChanged;

        private ObservableValue<PathObjectHierarchy> hierarchyProperty;

        private IntegerProperty numSelectedAnnotations = new SimpleIntegerProperty();
        private IntegerProperty numSelectedDetections = new SimpleIntegerProperty();

        private BooleanProperty annotationSelected = new SimpleBooleanProperty();

        SelectedObjectCounter(ObservableValue<ImageData<BufferedImage>> imageDataProperty) {
            this.imageDataProperty.bind(imageDataProperty);
            this.hierarchyProperty = createHierarchyBinding();
            hierarchyProperty.addListener((observable, oldValue, newValue) -> updateHierarchy(oldValue, newValue));
            updateHierarchy(null, hierarchyProperty.getValue());
        }

        private ObjectBinding<PathObjectHierarchy> createHierarchyBinding() {
            return Bindings.createObjectBinding(() -> {
                        var imageData = imageDataProperty.get();
                        return imageData == null ? null : imageData.getHierarchy();
                    },
                    imageDataProperty);
        }

        private void updateHierarchy(PathObjectHierarchy oldValue, PathObjectHierarchy newValue) {
            if (oldValue == newValue)
                return;
            if (oldValue != null)
                oldValue.getSelectionModel().removePathObjectSelectionListener(selectionListener);
            if (newValue != null)
                newValue.getSelectionModel().addPathObjectSelectionListener(selectionListener);
            updateSelectedObjectCounts();
        }

        private void selectedPathObjectChanged(PathObject pathObjectSelected, PathObject previousObject, Collection<PathObject> allSelected) {
            updateSelectedObjectCounts();
            annotationSelected.set(!annotationSelected.get()); // quick boolean trigger I can listen for to update bar graph
        }

        private void updateSelectedObjectCounts() {
            var hierarchy = hierarchyProperty.getValue();
            if (hierarchy == null) {
                numSelectedAnnotations.set(0);
                numSelectedDetections.set(0);
            } else {
                var selected = hierarchy.getSelectionModel().getSelectedObjects();
                numSelectedAnnotations.set((int) selected.stream().filter(p -> p.isAnnotation()).count());
                numSelectedDetections.set((int) selected.stream().filter(p -> p.isDetection()).count());
            }
        }

    }
}

