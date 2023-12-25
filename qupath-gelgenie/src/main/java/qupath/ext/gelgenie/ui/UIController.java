package qupath.ext.gelgenie.ui;

import ai.djl.MalformedModelException;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.TranslateException;
import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.beans.binding.BooleanBinding;
import javafx.beans.property.*;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.beans.binding.ObjectBinding;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.geometry.Side;
import javafx.scene.Node;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.scene.web.WebView;
import org.controlsfx.control.PopOver;
import org.controlsfx.control.action.Action;
import org.controlsfx.control.action.ActionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.gelgenie.graphics.EmbeddedBarChart;
import qupath.ext.gelgenie.models.GelGenieModel;
import qupath.ext.gelgenie.models.ModelInterfacing;
import qupath.ext.gelgenie.djl_processing.PytorchManager;
import qupath.ext.gelgenie.tools.BandSorter;
import qupath.ext.gelgenie.tools.ImageTools;
import qupath.ext.gelgenie.models.ModelRunner;
import qupath.lib.gui.QuPathGUI;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.tools.WebViews;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.events.PathObjectSelectionListener;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.plugins.workflow.DefaultScriptableWorkflowStep;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import org.commonmark.parser.Parser;
import org.commonmark.ext.front.matter.YamlFrontMatterExtension;
import org.commonmark.ext.front.matter.YamlFrontMatterVisitor;
import org.commonmark.renderer.html.HtmlRenderer;


import static qupath.lib.gui.tools.GuiTools.promptToSetActiveAnnotationProperties;
import static qupath.lib.scripting.QP.*;

/**
 * This is the main class that controls things happening in the GUI window.  Most actions starts from a function here.
 */
public class UIController {
    private static final Logger logger = LoggerFactory.getLogger(UIController.class);

    public QuPathGUI qupath;

    // These attributes are all linked to FXML elements in the GUI.
    @FXML
    private Button runButton;
    @FXML
    private Button tableButton;

    @FXML
    private Button downloadButton;
    @FXML
    private Button infoButton;
    @FXML
    private ToggleButton toggleBandNames;
    @FXML
    private ToggleButton toggleAnnotations;
    @FXML
    private ToggleButton toggleOverlayAnnotations;
    @FXML
    private ToggleButton toggleBrush;
    @FXML
    private ToggleButton toggleSelect;
    @FXML
    private ToggleButton toggleMove;

    @FXML
    private Button globalBackgroundSelector;
    @FXML
    private Button labelButton;
    @FXML
    private Button autoLabelButton;
    @FXML
    private Button classButton;
    @FXML
    private Button autoClassButton;

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
    private CheckBox enableRollingBackground;
    @FXML
    private CheckBox genTableOnSelectedBands;
    @FXML
    private CheckBox useDJLCheckBox;

    @FXML
    private Spinner<Integer> localSensitivity;
    @FXML
    private Spinner<Integer> maxHistoDisplay;

    @FXML
    private Spinner<Integer> rollingRadius;

    @FXML
    private BarChart<String, Number> bandChart;
    @FXML
    private ChoiceBox<GelGenieModel> modelChoiceBox;
    @FXML
    private ChoiceBox<String> deviceChoiceBox;
    @FXML
    private TabPane mainTabGroup;
    @FXML
    private Tab modelTab;

    private final ObjectProperty<ImageData<BufferedImage>> imageDataProperty = new SimpleObjectProperty<>();
    private final static ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");

    private final ObjectProperty<Boolean> pendingTask = new SimpleObjectProperty<>();

    private SelectedObjectCounter selectedObjectCounter;
    private BooleanBinding runButtonBinding;

    // these elements are for the info button popups
    private WebView infoWebView = WebViews.create(true);
    private PopOver infoPopover = new PopOver(infoWebView);

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
        getModelsPopulateList(); // sets up model dropdown menu
        configureDevicesList(); // sets up hardware devices for computational speed-up
        configureAdditionalPersistentSettings(); // sets up miscellaneous persistent settings
        configureTabGroup(); // sets up tab pane

        logger.info("GelGenie GUI loaded without errors");
    }

    /*
    This function sets up a listener that resizes the tab pane to match the content in the selected tab.
     */
    private void configureTabGroup(){
        // Set an event handler for tab selection changes
        mainTabGroup.getSelectionModel().selectedItemProperty().addListener((observable, oldTab, newTab) -> {
            if (newTab != null) {
                resizeTabPane(newTab);
            }
        });
        Platform.runLater(() -> resizeTabPane(modelTab)); // runs function once on the landing page
    }

    /*
    This is the main functionality for resizing the tab pane (based on the internal VBox height).
     */
    private void resizeTabPane(Tab selectedTab) {
        // Assuming that the content of the tab is VBox
        double newHeight = 0;
        for (Node titlePane : ((VBox) selectedTab.getContent()).getChildren()) {
                newHeight += titlePane.getBoundsInParent().getHeight();
        }
        mainTabGroup.setPrefHeight(newHeight + 40); // 40 is a buffer for tab headers and padding
    }

    /*
    This function enables the DJL buttons only if the PyTorch engine is available.
     */
    private void configureDevicesList() {
        useDJLCheckBox.setSelected(false);
        deviceChoiceBox.setDisable(!useDJLCheckBox.isSelected());
        if (useDJLCheckBox.isSelected()) {
            addDevices();
        }
        useDJLCheckBox.selectedProperty().addListener((observable, oldValue, newValue) -> {
            if (newValue) {
                if (!PytorchManager.hasPyTorchEngine()) {
                    Dialogs.showErrorMessage(resources.getString("title"), resources.getString("error.download-pytorch"));
                    useDJLCheckBox.setSelected(false);
                }
            }
            deviceChoiceBox.setDisable(!newValue);
            addDevices();
        });
        deviceChoiceBox.getSelectionModel().selectedItemProperty().addListener(
                (value, oldValue2, newValue2) -> GelGeniePrefs.deviceProperty().set(newValue2));
    }

    /*
    This function scans the user's hardware to see if any GPU or MPL devices are available to speed up model computation.
     */
    private void addDevices() {
        var availableDevices = PytorchManager.getAvailableDevices();
        deviceChoiceBox.getItems().setAll(availableDevices);
        var selected = GelGeniePrefs.deviceProperty().get();
        if (availableDevices.contains(selected)) {
            deviceChoiceBox.getSelectionModel().select(selected);
        } else {
            deviceChoiceBox.getSelectionModel().selectFirst();
        }
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
    }

    /*
     * Sets permanent properties for embedded barchart
     */
    private void configureBarChart() {
        bandChart.setBarGap(0);
        bandChart.setCategoryGap(0);
        bandChart.setLegendSide(Side.TOP);
        bandChart.setAnimated(false); // animation looks nice but it gets old quickly
        bandChart.getXAxis().setLabel("Pixel Intensity");
        bandChart.getYAxis().setLabel("Frequency");
    }

    /*
    Links additional settings to a persistent state that haven't been covered in other areas.
     */
    private void configureAdditionalPersistentSettings(){
        localSensitivity.getValueFactory().valueProperty().bindBidirectional(GelGeniePrefs.localCorrectionPixels());
        rollingRadius.getValueFactory().valueProperty().bindBidirectional(GelGeniePrefs.rollingRadius());
    }

    /**
     * Attaches checkboxes to extension persistent settings and links the model checkboxes together so that
     * when one is selected, the other must be off.
     */
    private void configureCheckBoxes() {
        useDJLCheckBox.selectedProperty().bindBidirectional(GelGeniePrefs.useDJLProperty());
        deletePreviousBands.selectedProperty().bindBidirectional(GelGeniePrefs.deletePreviousBandsProperty());
        enableGlobalBackground.selectedProperty().bindBidirectional(GelGeniePrefs.globalCorrectionProperty());
        enableLocalBackground.selectedProperty().bindBidirectional(GelGeniePrefs.localCorrectionProperty());
        enableRollingBackground.selectedProperty().bindBidirectional(GelGeniePrefs.rollingCorrectionProperty());

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
    private void configureDisplayToggleButtons() {
        var actions = qupath.getOverlayActions();
        var editing_actions = qupath.getToolManager();

        configureActionToggleButton(actions.SHOW_NAMES, toggleBandNames);
        configureActionToggleButton(actions.SHOW_ANNOTATIONS, toggleAnnotations);
        configureActionToggleButton(actions.FILL_ANNOTATIONS, toggleOverlayAnnotations);
        configureActionToggleButton(editing_actions.MOVE_TOOL, toggleMove);
        configureActionToggleButton(editing_actions.BRUSH_TOOL, toggleBrush);
        configureActionToggleButton(editing_actions.SELECTION_MODE, toggleSelect);
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

        runButtonBinding = Bindings.createBooleanBinding(
                () ->  imageDataProperty.isNull().or(pendingTask.isNotNull()).get() ||
                        modelChoiceBox.getSelectionModel().getSelectedItem() == null ||
                        !modelChoiceBox.getSelectionModel().getSelectedItem().isValid(),
                imageDataProperty,
                pendingTask,
                modelChoiceBox.getSelectionModel().selectedItemProperty()
        );
        runButton.disableProperty().bind(runButtonBinding);
        tableButton.disableProperty().bind(imageDataProperty.isNull().or(pendingTask.isNotNull()));
        labelButton.disableProperty().bind(imageDataProperty.isNull().or(pendingTask.isNotNull()));
        autoLabelButton.disableProperty().bind(imageDataProperty.isNull().or(pendingTask.isNotNull()));
        classButton.disableProperty().bind(imageDataProperty.isNull().or(pendingTask.isNotNull()));
        autoClassButton.disableProperty().bind(imageDataProperty.isNull().or(pendingTask.isNotNull()));

        // global background selector button needs to be disabled if no annotation is selected
        globalBackgroundSelector.disableProperty().bind(this.selectedObjectCounter.numSelectedAnnotations.isEqualTo(0));
    }

    /**
     * Populate the available models & configure the UI elements to select and download models.
     */
    private void getModelsPopulateList() {
        ModelInterfacing.GelGenieModelCollection models = ModelInterfacing.getModelCollection();
        modelChoiceBox.getItems().setAll(models.getModels().values());
        modelChoiceBox.setConverter(new ModelInterfacing.ModelStringConverter(models));
        modelChoiceBox.getSelectionModel().selectedItemProperty().addListener(
                (v, o, n) -> downloadButton.setDisable((n == null) || n.isValid()));
        modelChoiceBox.getSelectionModel().selectedItemProperty().addListener(
                (v, o, n) -> infoButton.setDisable((n == null) || !n.isValid()|| !checkFileExists(n.getReadmeFile())));
        modelChoiceBox.getSelectionModel().selectFirst();
    }
    private static boolean checkFileExists(File file) {
        return file != null && file.isFile();
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

        int index = 0;
        Collection<String> annotNames = new ArrayList<>();
        for (PathObject annot : annotations) {
            ImageServer<BufferedImage> server = imageData.getServer();
            double[] all_pixels = ImageTools.extractAnnotationPixels(annot, server); // extracts a list of pixels matching the specific selected annotation
            dataList.add(all_pixels);
            annotNames.add(annot.getName());
            index++;
            if (index >= maxHistoDisplay.getValue()) {
                break;
            }
        }

        ObservableList<XYChart.Series<String, Number>> allPlots = EmbeddedBarChart.plotHistogram(dataList,
                                                                                            40, annotNames);
        bandChart.getData().addAll(allPlots); // adds new data

    }

    /**
     * Downloads the selected model from HuggingFace.
     */
    public void downloadModel() {
        var model = modelChoiceBox.getSelectionModel().getSelectedItem();
        if (model == null) {
            return;
        }
        if (model.isValid()) {
            showModelAvailableNotification(model.getName());
            return;
        }
        ForkJoinPool.commonPool().execute(() -> {
            model.removeCache();
            showDownloadingModelNotification(model.getName());
            try {
                model.downloadModel();
            } catch (IOException e) {
                Dialogs.showErrorMessage(resources.getString("title"), resources.getString("error.downloading"));
                logger.error("Error downloading model", e);
                return;
            }
            showModelAvailableNotification(model.getName());
            downloadButton.setDisable(true);
            runButtonBinding.invalidate(); // fire an update to the binding, so the run button becomes available
            infoButton.setDisable(false);
        });
    }

    /**
     *  Displays the model information in a popup window (edited from the wsinfer extension).
     */
    public void presentModelInfo(){
        if (infoPopover.isShowing()) {
            infoPopover.hide();
            return;
        }
        GelGenieModel model = modelChoiceBox.getSelectionModel().getSelectedItem();
        var file = model.getReadmeFile();
        if (!checkFileExists(file)) {
            logger.warn("Readme file not available: {}", file);
            return;
        }
        try {
            var markdown = Files.readString(file.toPath());
            // Parse the initial markdown only, to extract any YAML front matter
            var parser = Parser.builder()
                    .extensions(
                            Arrays.asList(YamlFrontMatterExtension.create())
                    ).build();
            var doc = parser.parse(markdown);
            var visitor = new YamlFrontMatterVisitor();
            doc.accept(visitor);
            var metadata = visitor.getData();

            // If we have YAML metadata, remove from the start (since it renders weirdly) and append at the end
            if (!metadata.isEmpty()) {
                doc.getFirstChild().unlink();
                var sb = new StringBuilder();
                sb.append("----\n\n");
                sb.append("### Metadata\n\n");
                for (var entry : metadata.entrySet()) {
                    sb.append("\n* **").append(entry.getKey()).append("**: ").append(entry.getValue());
                }
                doc.appendChild(parser.parse(sb.toString()));
            }

            // If the markdown doesn't start with a title, pre-pending the model title & description (if available)
            if (!markdown.startsWith("#")) {
                var sb = new StringBuilder();
                sb.append("## ").append(model.getName()).append("\n\n");
                var description = model.getDescription();
                if (description != null && !description.isEmpty()) {
                    sb.append("_").append(description).append("_").append("\n\n");
                }
                sb.append("----\n\n");
                doc.prependChild(parser.parse(sb.toString()));
            }

            infoWebView.getEngine().loadContent(
                    HtmlRenderer.builder().build().render(doc));
            infoPopover.show(infoButton);
        } catch (IOException e) {
            logger.error("Error parsing readme file", e);
        }
    }


    /**
     * Runs when user attempts to download a model that is already available.
     *
     * @param modelName: Model keyname
     */
    private void showModelAvailableNotification(String modelName) {
        Dialogs.showPlainNotification(
                resources.getString("title"),
                String.format(resources.getString("ui.popup.model-available"), modelName));
    }

    /**
     * Runs when model is being downloaded.
     *
     * @param modelName: Model keyname
     */
    private void showDownloadingModelNotification(String modelName) {
        Dialogs.showPlainNotification(
                resources.getString("title"),
                String.format(resources.getString("ui.popup.model-downloading"), modelName));
    }

    private void showRunningModelNotification() {
        Dialogs.showInfoNotification(
                resources.getString("title"),
                resources.getString("ui.popup.model-running"));
    }

    private void showCompleteModelNotification() {
        Dialogs.showInfoNotification(
                resources.getString("title"),
                resources.getString("ui.popup.model-complete"));
    }

    /**
     * Helper class for storing user preferences for model inference.
     * @param runFullImagePref: Whether to run the model on the entire image or just the selected annotation.
     * @param deletePreviousBandsPref: Whether to delete all previous annotations after generating new ones.
     */
    private record ModelInferencePreferences(boolean runFullImagePref, boolean deletePreviousBandsPref) { }

    /**
     * Runs the segmentation model on the provided image or within the selected annotation,
     * generating annotations for each located band.
     */
    public void runBandInference() {

        ModelInferencePreferences inferencePrefs = new ModelInferencePreferences(runFullImage.isSelected(), deletePreviousBands.isSelected());

        showRunningModelNotification();
        pendingTask.set(true);
        ImageData<BufferedImage> imageData = getCurrentImageData();

        ForkJoinPool.commonPool().execute(() -> {
            Collection<PathObject> newBands;
            if (inferencePrefs.runFullImagePref()) { // runs model on entire image
                try {
                    newBands = ModelRunner.runFullImageInference(modelChoiceBox.getSelectionModel().getSelectedItem(),
                            useDJLCheckBox.isSelected(), imageData);
                    addInferenceToHistoryWorkflow(imageData,
                            modelChoiceBox.getSelectionModel().getSelectedItem().getName(), useDJLCheckBox.isSelected());
                } catch (IOException | MalformedModelException | ModelNotFoundException | TranslateException e) {
                    pendingTask.set(null);
                    runButtonBinding.invalidate(); // fire an update to the binding, so the run button becomes available
                    throw new RuntimeException(e);
                }
            } else { // runs model on data within selected annotation only
                try {
                    newBands = ModelRunner.runAnnotationInference(modelChoiceBox.getSelectionModel().getSelectedItem(),
                            useDJLCheckBox.isSelected(), imageData, getSelectedObject());

                } catch (IOException | MalformedModelException | TranslateException | ModelNotFoundException e) {
                    pendingTask.set(null);
                    runButtonBinding.invalidate(); // fire an update to the binding, so the run button becomes available
                    throw new RuntimeException(e);
                }
            }

            if (inferencePrefs.deletePreviousBandsPref()) { //removes all annotations before adding new ones
                ArrayList<PathObject> removables = new ArrayList<>();
                for (PathObject annot : getAnnotationObjects()) {
                    var pathClass = PathClass.getInstance("Gel Band");
                    if (annot.getPathClass() != null && annot.getPathClass().equals(pathClass)) {
                        removables.add(annot);
                    }
                }
                removeObjects(removables, false);
            }
            pendingTask.set(null);

            if (newBands == null) {
                Dialogs.showWarningNotification(resources.getString("title"), resources.getString("error.no-bands"));
                return;
            }
            for (PathObject annot : newBands) {
                annot.setPathClass(PathClass.fromString("Gel Band", 10709517));
                // can use this converter to select the integer color from an RGB code: http://www.shodor.org/~efarrow/trunk/html/rgbint.html
            }
            addObjects(newBands);
            BandSorter.LabelBands(newBands);
            addLabellingToHistoryWorkflow(imageData);
            showCompleteModelNotification();
        });
    }

    public void autoLabelBands(){
        Collection<PathObject> actionableAnnotations = new ArrayList<>();
        for (PathObject annot : getAnnotationObjects()) {
            if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Gel Band")) {
                actionableAnnotations.add(annot); // histogram should only activate on bands not other objects
            }
        }
        BandSorter.LabelBands(actionableAnnotations);
    }

    public void manualBandLabel(){
        promptToSetActiveAnnotationProperties(getCurrentHierarchy());
    }

    public void manualSetClass(){
        PathObject annot = getSelectedObject();
        PathClass gClass = PathClass.fromString("Gel Band", 10709517);
        annot.setPathClass(gClass);
    }

    public void classifyFreeAnnotations(){
        PathClass gClass = PathClass.fromString("Gel Band", 10709517);
        for (PathObject annot : getAnnotationObjects()) {
            if (annot.getPathClass() == null) {
                annot.setPathClass(gClass);
            }
        }
    }

    private static void addInferenceToHistoryWorkflow(ImageData<?> imageData, String modelName, boolean useDJL) {
        imageData.getHistoryWorkflow()
                .addStep(
                        new DefaultScriptableWorkflowStep(
                                resources.getString("workflow.inference"),
                                ModelRunner.class.getName() + ".runFullImageInferenceAndAddAnnotations(\""+modelName+"\","+useDJL+")"
                        ));
    }

    private static void addLabellingToHistoryWorkflow(ImageData<?> imageData) {
        imageData.getHistoryWorkflow()
                .addStep(
                        new DefaultScriptableWorkflowStep(
                                resources.getString("workflow.labelling"),
                                BandSorter.class.getName() + ".LabelBands()"
                        ));
    }

    private static void addDataComputeAndExportToHistoryWorkflow(ImageData<?> imageData, boolean globalCorrection,
                                                                 boolean localCorrection, boolean rollingCorrection,
                                                                 int localSensitivity, int rollingRadius) {
        imageData.getHistoryWorkflow()
                .addStep(
                        new DefaultScriptableWorkflowStep(
                                resources.getString("workflow.computeandexport"),
                                TableController.class.getName() +
                                        ".computeAndExportBandData("+globalCorrection+","+localCorrection+","+rollingCorrection+","+localSensitivity+","+rollingRadius+",\"OUTPUT FOLDER\",\"OUTPUT FILENAME OR NULL\")"
                        ));
    }

    /**
     * Generates a band data table when requested.  User preferences are also passed on to the table creator class.
     */
    public void populateTable() {

        ArrayList<PathObject> selectedBands = new ArrayList<>();
        if (genTableOnSelectedBands.isSelected()) {
            for (PathObject annot : getSelectedObjects()) {
                if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Gel Band")) {
                    selectedBands.add(annot);
                }
            }
        }

        TableRootCommand tableCommand = new TableRootCommand(qupath, "gelgenie_table",
                "Data Table", true, enableGlobalBackground.isSelected(),
                enableLocalBackground.isSelected(), enableRollingBackground.isSelected(),
                localSensitivity.getValue(), rollingRadius.getValue(), selectedBands);
        tableCommand.run();

        // adds scriptable command for later execution
        addDataComputeAndExportToHistoryWorkflow(getCurrentImageData(), enableGlobalBackground.isSelected(),
                                                enableLocalBackground.isSelected(), enableRollingBackground.isSelected(),
                                                localSensitivity.getValue(), rollingRadius.getValue());
    }

    /**
     * Sets selected annotation to be used as the global background for band analysis.
     */
    @FXML
    private void setGlobalBackgroundPatch() {
        PathObject annot = getSelectedObject();
        PathClass gbClass = PathClass.fromString("Global Background", 906200);
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

