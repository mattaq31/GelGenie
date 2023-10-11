package qupath.ext.gelgenie.ui;

import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.QuPathGUI;

import java.util.Objects;
import java.util.ResourceBundle;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.VBox;

import java.io.IOException;
import java.net.URL;

/**
Boilerplate function for generating a UI window and setting things for it to be resizable etc.
 */
public class ActivateUI implements Runnable{

    private static final Logger logger = LoggerFactory.getLogger(ActivateUI.class);
    private final QuPathGUI qupath;
    private final String ui_name;

    private String panel_name = "None";
    private Boolean resizable = false;
    private final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");
    private Stage stage;

    public ActivateUI(QuPathGUI qupath, String ui_name) { // Constructor here
        this.qupath = qupath;
        this.ui_name = ui_name;
    }

    public ActivateUI(QuPathGUI qupath, String ui_name, String panel_name) { // Constructor 2
        this.qupath = qupath;
        this.ui_name = ui_name;
        this.panel_name = panel_name;
    }

    public ActivateUI(QuPathGUI qupath, String ui_name, String panel_name, Boolean resizable) { // Constructor 3
        this.qupath = qupath;
        this.ui_name = ui_name;
        this.panel_name = panel_name;
        this.resizable = resizable;
    }
    @Override
    public void run() { // generates UI page
        if (stage == null) {
            try {
                stage = createStage();
            } catch (IOException e) {
                logger.error(e.getMessage(), e);
                return;
            }
        }
        stage.show();
    }
    private Stage createStage() throws IOException {

//        URL url = getClass().getResource("/qupath/ext/gelgenie/ui/" + ui_name + ".fxml");  - this used to be

        // It used to be necessary to fill in a full path to the FXML file, but now it seems to work with just the name.
        // Will remove if this continues to be stable.
        URL url = getClass().getResource(ui_name + ".fxml");

        if (url == null) { // this should never happen...
            throw new IOException("Cannot find URL for GelGenie FXML");
        }

        // We need to use the ExtensionClassLoader to load the FXML, since it's in a different module TODO: is this needed?
        var loader = new FXMLLoader(url, resources);
//        loader.setClassLoader(QuPathGUI.getExtensionClassLoader());

        Pane root = loader.load();

        // There's probably a better approach... but wrapping in a border pane
        // helped me get the resizing to behave
        BorderPane pane = new BorderPane(root);
        Scene scene = new Scene(pane);

        Stage stage = new Stage();
        stage.initOwner(qupath.getStage());
        if (Objects.equals(panel_name, "None")){
            stage.setTitle(resources.getString("title"));
        }
        else {
            stage.setTitle(resources.getString("title") + " " + panel_name);
        }
        stage.setScene(scene);
        stage.setResizable(resizable);

        root.heightProperty().addListener((v, o, n) -> stage.sizeToScene());

        return stage;
    }

}
