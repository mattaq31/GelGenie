package qupath.ext.gelgenie.ui;

import javafx.beans.property.SimpleObjectProperty;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.image.ImageView;
import qupath.ext.gelgenie.tools.ImageTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.tools.GuiTools;
import qupath.lib.gui.commands.SummaryMeasurementTableCommand;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.regions.RegionRequest;

import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;

import javafx.embed.swing.SwingFXUtils;
import qupath.lib.roi.RectangleROI;
import qupath.lib.regions.ImageRegion;
import qupath.fx.dialogs.FileChoosers;

import static qupath.lib.scripting.QP.*;

public class TableController {

    //Table
    @FXML
    private TableView<BandEntry> mainTable;

    @FXML
    private TableColumn<BandEntry, ImageView> thumbnailCol;
    @FXML
    private TableColumn<BandEntry, Integer> bandCol;
    @FXML
    private TableColumn<BandEntry, String> nameCol;
    @FXML
    private TableColumn<BandEntry, Double> pixelCol;
    @FXML
    private TableColumn<BandEntry, Double> meanCol;
    @FXML
    private TableColumn<BandEntry, Double> rawCol;
    @FXML
    private TableColumn<BandEntry, Double> localVolCol;
    @FXML
    private TableColumn<BandEntry, Double> globalVolCol;
    @FXML
    private TableColumn<BandEntry, Double> normVolCol;

    private ObservableList<BandEntry> bandData;

    public QuPathGUI qupath;

    public TableController() {
        this.qupath = QuPathGUI.getInstance();
    }

    @FXML
    private void initialize() {
        bandCol.setCellValueFactory(new PropertyValueFactory<BandEntry, Integer>("bandID"));
        nameCol.setCellValueFactory(new PropertyValueFactory<BandEntry, String>("bandName"));
        pixelCol.setCellValueFactory(new PropertyValueFactory<BandEntry, Double>("pixelCount"));
        meanCol.setCellValueFactory(new PropertyValueFactory<BandEntry, Double>("averageIntensity"));
        rawCol.setCellValueFactory(new PropertyValueFactory<BandEntry, Double>("rawVolume"));
        localVolCol.setCellValueFactory(new PropertyValueFactory<BandEntry, Double>("globalVolume"));
        globalVolCol.setCellValueFactory(new PropertyValueFactory<BandEntry, Double>("localVolume"));
        normVolCol.setCellValueFactory(new PropertyValueFactory<BandEntry, Double>("normVolume"));
        thumbnailCol.setCellValueFactory(new PropertyValueFactory<BandEntry, ImageView>("thumbnail"));
        meanCol.setCellFactory(tc -> new TableCell<BandEntry, Double>() { //TODO: extend this to all other cells
            @Override
            protected void updateItem(Double value, boolean empty) {
                super.updateItem(value, empty);
                if (empty) {
                    setText(null);
                } else {
                    setText(String.format("%.2f", value.floatValue()));
                }
            }
        });

        ImageData<BufferedImage> imageData = getCurrentImageData();
        ImageServer<BufferedImage> server = imageData.getServer();
        Collection<PathObject> annots = getAnnotationObjects();
        for (PathObject annot : annots) {

            double[] all_pixels = ImageTools.extractAnnotationPixels(annot, server); // extracts a list of pixels matching the specific selected annotation

//            def roi = new RectangleROI(cx-sizePixels/2, cy-sizePixels/2, sizePixels, sizePixels)
//            def request = RegionRequest.createInstance(server.getPath(), 1.0, roi)
            int padding = 10;
            int x1 = (int) annot.getROI().getBoundsX() - padding;
            int y1 = (int) annot.getROI().getBoundsY() - padding;
            int x2 = (int) Math.ceil(annot.getROI().getBoundsX() + annot.getROI().getBoundsWidth()) + padding;
            int y2 = (int) Math.ceil(annot.getROI().getBoundsY() + annot.getROI().getBoundsHeight()) + padding;

            ImageRegion thumbnailRegion = ImageRegion.createInstance(x1, y1,
                    x2 - x1, y2 - y1, annot.getROI().getZ(), annot.getROI().getT());

            RegionRequest request = RegionRequest.createInstance(server.getPath(), 1.0, thumbnailRegion);
            BufferedImage img;
            try {
                img = server.readRegion(request);
            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
            ImageView imviewer = new ImageView();
            imviewer.setImage(SwingFXUtils.toFXImage(img, null));
            BandEntry curr_band = new BandEntry(8, annot.getPathClass().toString(), all_pixels.length,
                    Arrays.stream(all_pixels).average().getAsDouble(), Arrays.stream(all_pixels).sum(),
                    5.0, 5.0, 5.0, imviewer);
            ObservableList<BandEntry> all_bands = mainTable.getItems();
            all_bands.add(curr_band);

            bandData = all_bands;
            mainTable.setItems(all_bands);
        }

        // settings setup
        mainTable.setPlaceholder(new Label("No gel band data to display"));

        TableView.TableViewSelectionModel<BandEntry> selectionModel = mainTable.getSelectionModel();
        selectionModel.setSelectionMode(SelectionMode.MULTIPLE);

//        bandCol.prefWidthProperty().bind(mainTable.widthProperty().multiply(0.05));
//        bandCol.setStyle( "-fx-alignment: CENTER-RIGHT;");

//        areaCol.prefWidthProperty().bind(mainTable.widthProperty().multiply(0.7));


//        BandEntry test_table = new BandEntry("TEST",5);

//        ObservableList<BandEntry> all_bands = mainTable.getItems();
//        all_bands.add(test_table);
//        mainTable.setItems(all_bands);
    }

    public void exportData() throws IOException {

        File fileOutput = FileChoosers.promptToSaveFile("Export image region", null,
                                                        FileChoosers.createExtensionFilter("Set CSV output filename", ".csv"));
        if (fileOutput == null)
            return;

        BufferedWriter br = new BufferedWriter(new FileWriter(fileOutput));

        br.write("Band Name, ID\n");

        for (BandEntry band : bandData) {
            String sb = band.getBandName() + "," + band.getBandID() + "\n";
            br.write(sb);
        }
        br.close();
    }
}