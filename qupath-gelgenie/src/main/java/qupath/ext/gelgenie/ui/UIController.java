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
import qupath.lib.common.ThreadTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.commands.Commands;
//import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathTileObject;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.objects.hierarchy.events.PathObjectSelectionListener;
import qupath.lib.plugins.workflow.DefaultScriptableWorkflowStep;

import java.awt.image.BufferedImage;
import java.util.Collection;
import java.util.Objects;
import java.util.ResourceBundle;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;


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

    private final static ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");

    private Stage measurementMapsStage;

    private final ExecutorService pool = Executors.newSingleThreadExecutor(ThreadTools.createThreadFactory("gelgenie", true));

    private final ObjectProperty<Task<?>> pendingTask = new SimpleObjectProperty<>();

    @FXML
    private void initialize() {
        logger.info("Initializing...");

        this.qupath = QuPathGUI.getInstance();
        this.imageDataProperty.bind(qupath.imageDataProperty());
//        configureDisplayToggleButtons();

    }

    private void configureActionToggleButton(Action action, ToggleButton button) {
        ActionUtils.configureButton(action, button);
        button.setContentDisplay(ContentDisplay.GRAPHIC_ONLY);
    }

//    private void configureDisplayToggleButtons() {
//        var actions = qupath.getDefaultActions();
//        configureActionToggleButton(actions.FILL_DETECTIONS, toggleDetectionFill);
//        configureActionToggleButton(actions.SHOW_DETECTIONS, toggleDetections);
//        configureActionToggleButton(actions.SHOW_ANNOTATIONS, toggleAnnotations);
//    }
}
