package qupath.ext.gelgenie.ui;

import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.QuPathGUI;

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
    private final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");
    private Stage stage;

    public ActivateUI(QuPathGUI qupath) { // Constructor here
        this.qupath = qupath;
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

        URL url = getClass().getResource("/qupath/ext/gelgenie/ui/gelgenie_control.fxml");
        if (url == null) { // this should never happen...
            throw new IOException("Cannot find URL for GelGenie FXML");
        }

        // We need to use the ExtensionClassLoader to load the FXML, since it's in a different module TODO: is this needed?
        var loader = new FXMLLoader(url, resources);
//        loader.setClassLoader(QuPathGUI.getExtensionClassLoader());
        VBox root = loader.load();

        // There's probably a better approach... but wrapping in a border pane
        // helped me get the resizing to behave
        BorderPane pane = new BorderPane(root);
        Scene scene = new Scene(pane);

        Stage stage = new Stage();
        stage.initOwner(qupath.getStage());
        stage.setTitle(resources.getString("title"));
        stage.setScene(scene);
        stage.setResizable(false);

        root.heightProperty().addListener((v, o, n) -> stage.sizeToScene());

        return stage;
    }

}
