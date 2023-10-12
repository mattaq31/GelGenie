package qupath.ext.gelgenie.ui;

import javafx.application.Platform;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;

import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.image.ImageView;
import qupath.ext.gelgenie.tools.ImageTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.regions.RegionRequest;

import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Objects;

import javafx.embed.swing.SwingFXUtils;
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

    private boolean globalCorrection = false;
    private boolean localCorrection = false;
    private int localSensitivity = 5;

    double global_mean = 0.0;

    public TableController() {
        this.qupath = QuPathGUI.getInstance();
    }

    @FXML
    private void initialize() throws Exception {
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

        Platform.runLater(() -> {
            if (globalCorrection) {
                try {
                    global_mean = calculateGlobalBackgroundAverage(server);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }

            for (PathObject annot : annots) {
                if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Gel Band")) {
                    double[] all_pixels = ImageTools.extractAnnotationPixels(annot, server); // extracts a list of pixels matching the specific selected annotation

                    int padding = 10;  // TODO: make this user adjustable
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

                    double pixel_average = Arrays.stream(all_pixels).average().getAsDouble();
                    double raw_volume = Arrays.stream(all_pixels).sum();

                    BandEntry curr_band = new BandEntry(8, annot.getPathClass().toString(), all_pixels.length,
                            pixel_average, raw_volume,
                            raw_volume - (global_mean*all_pixels.length),
                            5.0, 5.0, imviewer);  //TODO: remove global mean column completely when not enabled

                    ObservableList<BandEntry> all_bands = mainTable.getItems();
                    all_bands.add(curr_band);

                    bandData = all_bands;
                    mainTable.setItems(all_bands);
                }
            }

        });
        // settings setup
        mainTable.setPlaceholder(new Label("No gel band data to display"));
        TableView.TableViewSelectionModel<BandEntry> selectionModel = mainTable.getSelectionModel();
        selectionModel.setSelectionMode(SelectionMode.MULTIPLE);
    }

    private double calculateGlobalBackgroundAverage(ImageServer<BufferedImage> server) throws Exception {
        double global_mean = 0.0;
        Collection<PathObject> annots = getAnnotationObjects();
        for (PathObject annot : annots) {
            if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Global Background")) {
                double[] all_pixels = ImageTools.extractAnnotationPixels(annot, server);
                global_mean = Arrays.stream(all_pixels).average().getAsDouble();
            }
        }
        if (global_mean == 0.0) {
            throw new Exception("No annotation marked for global background correction");
        }
        return global_mean;
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

    public void setPreferences(boolean globalCorrection, boolean localCorrection, int localSensitivity) {
        this.localCorrection = localCorrection;
        this.globalCorrection = globalCorrection;
        this.localSensitivity = localSensitivity;
    }

}