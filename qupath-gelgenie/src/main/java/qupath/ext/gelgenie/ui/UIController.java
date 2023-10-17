package qupath.ext.gelgenie.ui;

import javafx.beans.binding.Bindings;
import javafx.beans.property.*;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.beans.binding.ObjectBinding;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.collections.ObservableList;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import org.controlsfx.control.action.Action;
import org.controlsfx.control.action.ActionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.gelgenie.graphics.EmbeddedBarChart;
import qupath.ext.gelgenie.tools.ImageTools;
import qupath.ext.gelgenie.tools.openCVModelRunner;
import qupath.lib.common.ThreadTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.events.PathObjectSelectionListener;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.objects.DefaultPathObjectComparator;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import static qupath.lib.scripting.QP.*;

/**
 * This is the main class that controls things happening in the GUI window.  Most actions starts from a function here.
 */
public class UIController {
    private static final Logger logger = LoggerFactory.getLogger(UIController.class);

    public QuPathGUI qupath;

    // These attributes are all linked to FXML elements in the GUI.
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
    private Button globalBackgroundSelector;

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
    private CheckBox genTableOnSelectedBands;

    @FXML
    private Spinner<Integer> localSensitivity;
    @FXML
    private BarChart<String, Number> bandChart;
    @FXML
    private ChoiceBox<String> modelChoiceBox;

    private final ObjectProperty<ImageData<BufferedImage>> imageDataProperty = new SimpleObjectProperty<>();
    private final static ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");

    private final ExecutorService pool = Executors.newSingleThreadExecutor(ThreadTools.createThreadFactory("gelgenie", true));

    private final ObjectProperty<Task<?>> pendingTask = new SimpleObjectProperty<>();

    private SelectedObjectCounter selectedObjectCounter;
    private int bandIdCounter = 0;

    /*
    This function is the first to run when the GUI window is created.
     */
    @FXML
    private void initialize() {
        logger.info("Initializing GelGenie GUI");

        this.qupath = QuPathGUI.getInstance(); // linking to QuPath Instance
        this.imageDataProperty.bind(qupath.imageDataProperty());
        this.selectedObjectCounter = new SelectedObjectCounter(this.imageDataProperty);  // main listener that tracks selected annotations and changes in selection

        configureDisplayToggleButtons(); // styles QuPath linked buttons
        configureButtonInteractivity(); // sets rules for visibility of certain buttons
        configureCheckBoxes(); // sets rules for checkboxes
        configureBarChart(); // sets up embedded bar chart
        configureAnnotationListener(); // sets up listener that triggers the appropriate functions when an annotation is selected/deselected

        modelChoiceBox.getItems().add("Prototype-UNet-July-29-2023"); // current list of models hardcoded, will update in future.
        modelChoiceBox.setValue("Prototype-UNet-July-29-2023");

        logger.info("GelGenie GUI loaded without errors");
    }

    /*
     * This sets up a listener to update the GUI histogram whenever a user selects/deselects an annotation.
     */
    private void configureAnnotationListener() {
        ChangeListener<Boolean> changeListener = (observable, oldValue, newValue) -> {
            if (selectedObjectCounter.numSelectedAnnotations.get() != 0) { // only trigger when annotations are selected
                Collection<PathObject> actionableAnnotations = new ArrayList<>();
                for (PathObject annot : getSelectedObjects()) {
                    if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Gel Band")) {
                        actionableAnnotations.add(annot); // histogram should only activate on bands not other objects
                    }
                }
                if (!actionableAnnotations.isEmpty() && actionableAnnotations.size() < 20) { // TODO: make this user-definable
                    BandHistoDisplay(actionableAnnotations);
                } else {
                    bandChart.getData().clear();
                }

            } else {
                bandChart.getData().clear();
            }
        };
        selectedObjectCounter.annotationSelected.addListener(changeListener);
    }

    /*
     * Sets permanent properties for embedded barchart
     */
    private void configureBarChart() {
        bandChart.setBarGap(0);
        bandChart.setCategoryGap(0);
        bandChart.setLegendVisible(false);
        bandChart.setAnimated(false); // todo: could consider making this less annoying...
        bandChart.getXAxis().setLabel("Pixel Intensity");
        bandChart.getYAxis().setLabel("Frequency");
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
     * There are currently only 2 for GelGenie.
     */
    private void configureDisplayToggleButtons() {
        var actions = qupath.getOverlayActions();
        configureActionToggleButton(actions.SHOW_NAMES, toggleBandNames);
        configureActionToggleButton(actions.SHOW_ANNOTATIONS, toggleAnnotations);
    }

    /**
     * Links buttons with their respective graphics from the general QuPath interface.
     */
    private void configureActionToggleButton(Action action, ToggleButton button) {
        ActionUtils.configureButton(action, button);
        button.setContentDisplay(ContentDisplay.GRAPHIC_ONLY);
    }

    /**
     * Links button availability according to availability of images, annotations, etc.
     */
    private void configureButtonInteractivity() {
        // TODO: pending task property currently unused.
        runButton.disableProperty().bind(imageDataProperty.isNull().or(pendingTask.isNotNull()));
        tableButton.disableProperty().bind(imageDataProperty.isNull().or(pendingTask.isNotNull()));

        // global background selector button needs to be disabled if no annotation is selected
        globalBackgroundSelector.disableProperty().bind(this.selectedObjectCounter.numSelectedAnnotations.isEqualTo(0));
    }

    /**
     * Updates the live histogram with the distributions of the selected gel bands.
     *
     * @param annotations: Collection of annotations to be processed and displayed.
     */
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
     * Runs the segmentation model on the provided image or within the selected annotation,
     * generating annotations for each located band.
     *
     * @throws IOException
     */
    public void runBandInference() throws IOException {
        ImageData<BufferedImage> imageData = getCurrentImageData();
        openCVModelRunner modelRunner = new openCVModelRunner("Prototype-UNet-July-29-2023"); //todo: remove hardcoding

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
            bandIdCounter = 0; // resets naming scheme
        }

        assert newBands != null; // todo: is this enough - what to show user if nothing found?

        ArrayList<PathObject> castBands = (ArrayList<PathObject>) newBands; // todo: how to remove this?

        // todo: this sorting isn't great, but will do for now.  Ideally the numbering would be sorted per lane, but have no way to detect lanes for now.
        castBands.sort(Comparator.comparing((PathObject p) -> p.getROI().getCentroidY()).thenComparing(p -> p.getROI().getCentroidX()));

        for (PathObject annot : castBands) {
            bandIdCounter++;
            annot.setPathClass(PathClass.fromString("Gel Band", 8000));
            annot.setName(String.valueOf(bandIdCounter));
        }

        addObjects(castBands);
    }

    /**
     * Generates a band data table when requested.  User preferences are also passed on to the table creator class.
     */
    public void populateTable() {

        Collection<PathObject> selectedBands = new ArrayList<>();
        if(genTableOnSelectedBands.isSelected()) {
            for (PathObject annot : getSelectedObjects()) {
                if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Gel Band")) {
                    selectedBands.add(annot);
                }
            }
        }
        TableRootCommand tableCommand = new TableRootCommand(qupath, "gelgenie_table",
                "Data Table", true, enableGlobalBackground.isSelected(),
                enableLocalBackground.isSelected(), localSensitivity.getValue(), selectedBands);
        tableCommand.run();
    }

    /**
     * Sets selected annotation to be used as the global background for band analysis.
     */
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

