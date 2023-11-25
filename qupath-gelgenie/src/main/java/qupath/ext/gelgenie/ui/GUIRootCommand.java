package qupath.ext.gelgenie.ui;

import javafx.scene.image.Image;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.ExtensionClassLoader;
import qupath.lib.gui.QuPathGUI;

import java.util.Objects;
import java.util.ResourceBundle;

import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;

import java.io.IOException;
import java.net.URL;

/**
 * Sets up main GUI window, which is fixed in size.
 */
public class GUIRootCommand implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(GUIRootCommand.class);
    private final QuPathGUI qupath;
    private final String ui_name;

    private final String panel_name;
    private final Boolean resizable;
    private final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");
    private Stage stage;

    public GUIRootCommand(QuPathGUI qupath, String ui_name, String panel_name, Boolean resizable) { // Constructor
        this.qupath = qupath;
        this.ui_name = ui_name;
        this.panel_name = panel_name;
        this.resizable = resizable;
    }

    @Override
    public void run() {
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

        URL url = getClass().getResource(ui_name + ".fxml");

        var loader = new FXMLLoader(url, resources);

        loader.setClassLoader(ExtensionClassLoader.getInstance()); // this won't work in QuPath v0.4

        Pane root = loader.load();

        // There's probably a better approach... but wrapping in a border pane
        // helped me get the resizing to behave - TODO: is this needed?
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

        // TODO: when github repo is live, replace the local logo with the online github one (or cache?)
        // TODO: create an 'About' page with bigger logo inside it + developer names
        // stage.getIcons().add(new Image("file:/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/graphics/logo/v1/gelgenie_small.png"));
        stage.setScene(scene);
        stage.setResizable(resizable);

        root.heightProperty().addListener((v, o, n) -> stage.sizeToScene());

        return stage;
    }

}
