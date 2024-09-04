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

package qupath.ext.gelgenie.ui;

import ai.djl.MalformedModelException;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.TranslateException;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import javafx.application.Platform;
import javafx.beans.InvalidationListener;
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
import javafx.geometry.Pos;
import javafx.geometry.Side;
import javafx.scene.Node;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Text;
import javafx.scene.text.TextAlignment;
import javafx.scene.text.TextFlow;
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
import qupath.ext.gelgenie.tools.SegmentationMap;
import qupath.fx.dialogs.FileChoosers;
import qupath.imagej.tools.IJTools;
import qupath.lib.common.GeneralTools;
import qupath.lib.gui.QuPathGUI;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.tools.WebViews;
import qupath.lib.images.ImageData;
import qupath.lib.images.PathImage;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ServerTools;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.events.PathObjectSelectionListener;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.plugins.workflow.DefaultScriptableWorkflowStep;

import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;

import org.commonmark.parser.Parser;
import org.commonmark.ext.front.matter.YamlFrontMatterExtension;
import org.commonmark.ext.front.matter.YamlFrontMatterVisitor;
import org.commonmark.renderer.html.HtmlRenderer;
import qupath.lib.regions.RegionRequest;
import qupath.lib.scripting.QP;


import static qupath.lib.gui.tools.GuiTools.promptToSetActiveAnnotationProperties;
import static qupath.lib.scripting.QP.*;

/**
 * This is the main class that controls things happening in the GUI window.  Most actions starts from a function here.
 */
public class UIController {
    private static final Logger logger = LoggerFactory.getLogger(UIController.class);

    public QuPathGUI qupath;

    // These attributes are all linked to FXML elements in the GUI.
    // BUTTONS HERE
    @FXML
    private Button runButton;
    @FXML
    private Button tableButton;
    @FXML
    private Button downloadButton;
    @FXML
    private Button infoButton;
    @FXML
    private Button infoButtonLabelEdit;
    @FXML
    private Button infoButtonAutoLabel;
    @FXML
    private Button infoButtonGlobalBg;
    @FXML
    private Button infoButtonLocalBg;
    @FXML
    private Button infoButtonRollingBg;
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
    private Button exportMapButton;

    // TOGGLE BUTTONS HERE
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

    // CHECKBOXES HERE
    @FXML
    private CheckBox runFullImage;
    @FXML
    private CheckBox runSelected;
    @FXML
    private CheckBox deletePreviousBands;
    @FXML
    private CheckBox imageInversion;
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

    // SPINNERS HERE
    @FXML
    private Spinner<Integer> localSensitivity;
    @FXML
    private Spinner<Integer> maxHistoDisplay;
    @FXML
    private Spinner<Integer> rollingRadius;

    // INFO POPUPS HERE
    private WebView infoWebView = WebViews.create(true);
    private PopOver infoPopover = new PopOver(infoWebView);
    private WebView labelInfoWebView = WebViews.create(true);
    private PopOver labelInfoPopover = new PopOver(labelInfoWebView);
    private WebView autoLabelInfoWebView = WebViews.create(true);
    private PopOver autoLabelInfoPopover = new PopOver(autoLabelInfoWebView);
    private WebView globalInfoWebView = WebViews.create(true);
    private PopOver globalInfoPopover = new PopOver(globalInfoWebView);
    private WebView localInfoWebView = WebViews.create(true);
    private PopOver localInfoPopover = new PopOver(localInfoWebView);
    private WebView rollingInfoWebView = WebViews.create(true);
    private PopOver rollingInfoPopover = new PopOver(rollingInfoWebView);

    // MISC HERE
    @FXML
    private BarChart<String, Number> bandChart;
    @FXML
    private ComboBox<GelGenieModel> modelChoiceBox;
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

    private static boolean checkFileExists(File file) {
        return file != null && file.isFile();
    }

    /*
    This function is the first to run when the GUI window is created - it prepares the whole interface for use.
     */
    @FXML
    private void initialize() throws IOException {
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

        if(qupath.getImageData() != null){
            imageInversion.setSelected(!checkGelImageInversion());
        }

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

                    // constructs an error message dialog which links to the QuPath DJL extension page
                    // to allow users to download the PyTorch engine if this is not already available
                    var linkDJL = new Hyperlink();
                    linkDJL.setText(resources.getString("error.download-pytorch-link"));
                    linkDJL.setOnAction(e -> QuPathGUI.openInBrowser(resources.getString("error.download-pytorch-link")));
                    var dialogText = new TextFlow( // mix of strings and a hyperlink
                            new Text(resources.getString("error.download-pytorch-1")),
                            linkDJL,
                            new Text(System.lineSeparator()),
                            new Text(resources.getString("error.download-pytorch-2"))
                    );
                    dialogText.setPrefWidth(500); // to accommodate the length of the link
                    dialogText.setTextAlignment(TextAlignment.CENTER);

                    new Dialogs.Builder()
                            .alertType(Alert.AlertType.ERROR)
                            .title(resources.getString("title"))
                            .content(dialogText)
                            .show();

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

        // checks the image to see if it's a dark or light background and then sets GUI checkbox accordingly.
        // The checkbox can be manually adjusted by the user TODO: can this value be retained and associated with the specific image that's open?
        qupath.imageDataProperty().addListener((observable, oldValue, newValue) -> {
            if (getCurrentImageData() != null) {
                try {
                    imageInversion.setSelected(!checkGelImageInversion());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        });

    }

    /*
      Links additional settings to a persistent state that haven't been covered in other areas.
       */
    private void configureAdditionalPersistentSettings(){
        localSensitivity.getValueFactory().valueProperty().bindBidirectional(GelGeniePrefs.localCorrectionPixels());
        rollingRadius.getValueFactory().valueProperty().bindBidirectional(GelGeniePrefs.rollingRadius());
    }

    /**
     * Configures the graphic and functionality of QuPath-linked toggle buttons.
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
        exportMapButton.disableProperty().bind(imageDataProperty.isNull().or(pendingTask.isNotNull()));
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

        // Prepares headers to visually sort the many models available from HuggingFace
        // Primary models are those that work best with most images
        GelGenieModel primaryDummyHeader = new GelGenieModel();
        primaryDummyHeader.setAbbrvName("Primary Models");
        primaryDummyHeader.setDummyModel(true); // not a true model - just a placeholder to act as a header

        GelGenieModel prototypeDummyHeader = new GelGenieModel();
        prototypeDummyHeader.setAbbrvName("Prototype Models");
        prototypeDummyHeader.setDummyModel(true);

        models.addModel("Dummy Header - Primary", primaryDummyHeader); // adds to the list of models
        models.addModel("Dummy Header - Prototypes", prototypeDummyHeader);

        // the below arranges the models in the dropdown menu in a more logical order - is it possible to make this more elegant?
        Collection<GelGenieModel> default_model_order = models.getModels().values();
        Collection<GelGenieModel> header_arranged_order = new ArrayList<>();

        header_arranged_order.add(primaryDummyHeader);
        for (GelGenieModel model:default_model_order) {
            if (!model.isDummyModel() && model.getModelType().equals("Primary")) {
                header_arranged_order.add(model);
            }
        }
        header_arranged_order.add(prototypeDummyHeader);
        for (GelGenieModel model:default_model_order) {
            if (!model.isDummyModel() && model.getModelType().equals("Prototype")) {
                header_arranged_order.add(model);
            }
        }

        // Customize the ListCell to turn dummy models into headers - centered + bold
        modelChoiceBox.setCellFactory(cb -> new ListCell<GelGenieModel>() {
            @Override
            protected void updateItem(GelGenieModel item, boolean empty) {
                super.updateItem(item, empty);
                if (empty) {
                    setText(null);
                } else {
                    setText(item.getAbbrvName());
                    setDisable(false);
                    setAlignment(Pos.CENTER_LEFT);
                    setStyle("");
                    if (item.isDummyModel()) {
                        setDisable(true); // Disable this item (headers should never be selected)
                        setAlignment(Pos.CENTER);
                        setStyle("-fx-font-weight: bold;");
                    }
                }
            }
        });

        modelChoiceBox.getItems().setAll(header_arranged_order);
        modelChoiceBox.setConverter(new ModelInterfacing.ModelStringConverter(models));

        modelChoiceBox.getSelectionModel().selectedItemProperty().addListener(
                (v, o, n) -> downloadButton.setDisable((n == null) || n.isValid()));
        modelChoiceBox.getSelectionModel().selectedItemProperty().addListener(
                (v, o, n) -> infoButton.setDisable((n == null) || !n.isValid()|| !checkFileExists(n.getReadmeFile())));
        modelChoiceBox.getSelectionModel().select(models.getModels().get("GelGenie-Universal-Dec-2023")); // default should always be the universal model
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
     * Reads a markdown file from the internal resources folder.  Based on this stackoverflow question:
     * https://stackoverflow.com/questions/6068197/read-resource-text-file-to-string-in-java
     * @param fileName: Name of the markdown file to be read.
     * @return File converted into a string.
     * @throws IOException
     */
    private String getResourceFileAsString(String fileName) throws IOException {
        try (InputStream is = getClass().getResourceAsStream(fileName)) {
            if (is == null) return null;
            try (InputStreamReader isr = new InputStreamReader(is);
                 BufferedReader reader = new BufferedReader(isr)) {
                return reader.lines().collect(Collectors.joining(System.lineSeparator()));
            }
        }
    }

    /**
     * Shows an info pop-up from a markdown file read from the resources folder.
     * @param popover: Popover object that controls the pop-up window.
     * @param webView: WebView object that displays the markdown file.
     * @param button: Button object that triggers the pop-up window.
     * @param fileName: Name of the markdown file to be displayed.
     * @param height: The max height of the webview popup.
     * @throws IOException
     */
    private void showInternalToolTip(PopOver popover, WebView webView, Button button, String fileName, int height) throws IOException {
        if (popover.isShowing()) {
            popover.hide();
            return;
        }
        var parser = Parser.builder().build();
        var doc = parser.parse(getResourceFileAsString(fileName));
        webView.getEngine().loadContent(HtmlRenderer.builder().build().render(doc));
        webView.setPrefHeight(height);
        popover.show(button);
    }

    // ALL TOOLTIP FUNCTIONS HERE
    public void presentManualLabellingTooltip() throws IOException {
        showInternalToolTip(labelInfoPopover, labelInfoWebView, infoButtonLabelEdit, "band_edit_help.md", 400);
    }

    public void presentAutoLabellingTooltip() throws IOException {
        showInternalToolTip(autoLabelInfoPopover, autoLabelInfoWebView, infoButtonAutoLabel, "auto_band_edit_help.md", 220);
    }

    public void presentGlobalBgTooltip() throws IOException {
        showInternalToolTip(globalInfoPopover, globalInfoWebView, infoButtonGlobalBg, "global_background_help.md", 285);
    }

    public void presentRollingBgTooltip() throws IOException {
        showInternalToolTip(rollingInfoPopover, rollingInfoWebView, infoButtonRollingBg, "rolling_background_help.md", 170);
    }

    public void presentLocalBgTooltip() throws IOException {
        showInternalToolTip(localInfoPopover, localInfoWebView, infoButtonLocalBg, "local_background_help.md", 205);
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
     * Checks the image and attempts to automatically determine if the image is inverted (dark bands on light background).  Can make mistakes in some cases if difference not so obvious.
     * @return True if the image is inverted, false if not.
     * @throws IOException
     */
    public boolean checkGelImageInversion() throws IOException {
        ImageServer<BufferedImage> server = getCurrentImageData().getServer();
        double max_val = getCurrentImageData().getServer().getPixelType().getUpperBound().doubleValue();
        // Use the entire image at full resolution
        RegionRequest request = RegionRequest.createInstance(server, 1.0);
        PathImage pathImage = IJTools.convertToImagePlus(server, request);
        ImagePlus imagePlus = (ImagePlus) pathImage.getImage();// Convert PathImage into ImagePlus
        ImageProcessor ip = imagePlus.getProcessor(); // Get ImageProcessor from ImagePlus
        ip.setAutoThreshold("Otsu"); // get estimated background threshold from image using Otsu's method
        double thresholdValue = ip.getMaxThreshold();

        if (thresholdValue > max_val/2) {
            // not very intelligent,
            // but if threshold reported by Otsu's method is
            // higher than half the dynamic range, this is generally an inverted image.
            // User can manually change this value if incorrect.
            return true;
        } else {
            return false;
        }
    }

    /**
     * Helper class for storing user preferences for model inference.
     * @param runFullImagePref: Whether to run the model on the entire image or just the selected annotation.
     * @param deletePreviousBandsPref: Whether to delete all previous annotations after generating new ones.
     * @param invertedImage: Whether the image is inverted (i.e. dark bands on light background) or not.
     */
    private record ModelInferencePreferences(boolean runFullImagePref, boolean deletePreviousBandsPref,
                                             boolean invertedImage) { }

    /**
     * Runs the segmentation model on the provided image or within the selected annotation,
     * generating annotations for each located band.
     */
    public void runBandInference() {

        ModelInferencePreferences inferencePrefs = new ModelInferencePreferences(runFullImage.isSelected(),
                deletePreviousBands.isSelected(), !imageInversion.isSelected());

        showRunningModelNotification();
        pendingTask.set(true);
        ImageData<BufferedImage> imageData = getCurrentImageData();

        ForkJoinPool.commonPool().execute(() -> {
            Collection<PathObject> newBands;
            if (inferencePrefs.runFullImagePref()) { // runs model on entire image
                try {
                    newBands = ModelRunner.runFullImageInference(modelChoiceBox.getSelectionModel().getSelectedItem(),
                            useDJLCheckBox.isSelected(), inferencePrefs.invertedImage(), imageData);
                    addInferenceToHistoryWorkflow(imageData,
                            modelChoiceBox.getSelectionModel().getSelectedItem().getName(), useDJLCheckBox.isSelected(), inferencePrefs.invertedImage());
                } catch (IOException | MalformedModelException | ModelNotFoundException | TranslateException e) {
                    pendingTask.set(null);
                    runButtonBinding.invalidate(); // fire an update to the binding, so the run button becomes available
                    throw new RuntimeException(e);
                }
            } else { // runs model on data within selected annotation only
                try {
                    newBands = ModelRunner.runAnnotationInference(modelChoiceBox.getSelectionModel().getSelectedItem(),
                            useDJLCheckBox.isSelected(), inferencePrefs.invertedImage(), imageData, getSelectedObject());

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
                // TODO: add this to script too (or make it optional)
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

    /**
     * Brings up the annotation properties window for the selected annotation.
     */
    public void manualBandLabel(){
        promptToSetActiveAnnotationProperties(getCurrentHierarchy());
    }

    /**
     * Marks the selected annotations as a gel band.
     */
    public void manualSetClass(){
        PathClass gClass = PathClass.fromString("Gel Band", 10709517);
        for (PathObject annot : getSelectedObjects()) {
            annot.setPathClass(gClass);
        }
    }

    /**
     * Automatically labels all gel bands in the image.
     */
    public void autoLabelBands(){
        Collection<PathObject> actionableAnnotations = new ArrayList<>();
        for (PathObject annot : getAnnotationObjects()) {
            if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Gel Band")) {
                actionableAnnotations.add(annot); // histogram should only activate on bands not other objects
            }
        }
        BandSorter.LabelBands(actionableAnnotations);
        fireHierarchyUpdate();
    }

    /**
     * Converts all unclassified annotations to gel bands.
     */
    public void classifyFreeAnnotations(){

        PathClass gClass = PathClass.fromString("Gel Band", 10709517);
        for (PathObject annot : getAnnotationObjects()) {
            if (annot.getPathClass() == null) {
                annot.setPathClass(gClass);
            }
        }
        fireHierarchyUpdate();
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
                localSensitivity.getValue(), rollingRadius.getValue(), !imageInversion.isSelected(), selectedBands);
        tableCommand.run();

        // adds scriptable command for later execution
        addDataComputeAndExportToHistoryWorkflow(getCurrentImageData(), enableGlobalBackground.isSelected(),
                                                enableLocalBackground.isSelected(), enableRollingBackground.isSelected(),
                                                localSensitivity.getValue(), rollingRadius.getValue(), !imageInversion.isSelected());
    }

    /**
     * Sets selected annotation to be used as the global background for band analysis.
     */
    public void setGlobalBackgroundPatch() {
        PathObject annot = getSelectedObject();
        PathClass gbClass = PathClass.fromString("Global Background", 906200);
        annot.setPathClass(gbClass);
    }

    /**
     * Exports annotations in image into a segmentation map which can be used for training models.
     * @throws IOException
     */
    public void exportSegmentationMap() throws IOException {

        // asks user for output filename
        String defaultName = GeneralTools.getNameWithoutExtension(new File(ServerTools.getDisplayableImageName(getCurrentImageData().getServer())));
        File fileOutput = FileChoosers.promptToSaveFile("Export Segmentation Map",  new File(defaultName + "_segmap.tif"),
                FileChoosers.createExtensionFilter("Save as TIF", ".tif"));

        // preps image for export
        SegmentationMap.exportSegmentationMap(fileOutput.toString());

        // records export command for scripting
        addSegmentationExportToHistoryWorkflow(getCurrentImageData());
    }

    // WORKFLOW RECORDERS HERE
    private static void addSegmentationExportToHistoryWorkflow(ImageData<?> imageData) {
        imageData.getHistoryWorkflow()
                .addStep(
                        new DefaultScriptableWorkflowStep(
                                resources.getString("workflow.exportsegmentation"),
                                SegmentationMap.class.getName() + ".exportSegmentationMapToProjectFolder()"
                        ));
    }

    private static void addInferenceToHistoryWorkflow(ImageData<?> imageData, String modelName, boolean useDJL, boolean invertImage) {
        imageData.getHistoryWorkflow()
                .addStep(
                        new DefaultScriptableWorkflowStep(
                                resources.getString("workflow.inference"),
                                ModelRunner.class.getName() + ".runFullImageInferenceAndAddAnnotations(\""+modelName+"\","+useDJL+","+invertImage+")"
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
                                                                 int localSensitivity, int rollingRadius, boolean invertImage) {
        imageData.getHistoryWorkflow()
                .addStep(
                        new DefaultScriptableWorkflowStep(
                                resources.getString("workflow.computeandexport"),
                                TableController.class.getName() +
                                        ".computeAndExportBandData("+globalCorrection+","+localCorrection+","+rollingCorrection+",\"Global\","+localSensitivity+","+rollingRadius+","+invertImage+",\"OUTPUT FOLDER\",\"OUTPUT FILENAME OR NULL\")"
                        ));
    }

    // NOTIFICATIONS HERE
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
            double[] all_pixels = ImageTools.extractAnnotationPixels(annot, server, !imageInversion.isSelected()); // extracts a list of pixels matching the specific selected annotation
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

