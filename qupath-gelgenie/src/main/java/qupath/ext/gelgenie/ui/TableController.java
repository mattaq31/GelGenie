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

import static qupath.ext.gelgenie.tools.ImageTools.createAnnotationImageFrame;
import static qupath.ext.gelgenie.tools.ImageTools.extractLocalBackgroundPixels;
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

    double globalMean = 0.0;

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
        localVolCol.setCellValueFactory(new PropertyValueFactory<BandEntry, Double>("localVolume"));
        globalVolCol.setCellValueFactory(new PropertyValueFactory<BandEntry, Double>("globalVolume"));
        normVolCol.setCellValueFactory(new PropertyValueFactory<BandEntry, Double>("normVolume"));
        thumbnailCol.setCellValueFactory(new PropertyValueFactory<BandEntry, ImageView>("thumbnail"));
        meanCol.setCellFactory(tc -> new TableCell<BandEntry, Double>() { //TODO: how do I combine these into one function?
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

        localVolCol.setCellFactory(tc -> new TableCell<BandEntry, Double>() {
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

        globalVolCol.setCellFactory(tc -> new TableCell<BandEntry, Double>() {
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
                    globalMean = calculateGlobalBackgroundAverage(server);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }

            for (PathObject annot : annots) {
                if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Gel Band")) {
                    double[] all_pixels = ImageTools.extractAnnotationPixels(annot, server); // extracts a list of pixels matching the specific selected annotation

                    int padding = 10;  // TODO: make this user adjustable

                    ImageRegion thumbnailRegion = createAnnotationImageFrame(annot, padding);
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

                    // todo: does it make sense to mask all bands rather than just the selected one?
                    double[] localBackgroundPixels = extractLocalBackgroundPixels(annot, server, localSensitivity);
                    double localMean = Arrays.stream(localBackgroundPixels).average().getAsDouble();

                    double globalVolume = 0;
                    double localVolume = 0;

                    if (globalCorrection){
                        globalVolume = raw_volume - (globalMean * all_pixels.length);
                    }
                    if (localCorrection){
                        localVolume = raw_volume - (localMean * all_pixels.length);
                    }

                    BandEntry curr_band = new BandEntry(8, annot.getPathClass().toString(), all_pixels.length,
                            pixel_average, raw_volume, globalVolume, localVolume, 5.0, imviewer);  //TODO: remove global/local mean column completely when not enabled

                    ObservableList<BandEntry> all_bands = mainTable.getItems();
                    all_bands.add(curr_band);

                    bandData = all_bands;
                    mainTable.setItems(all_bands);
                    if (!localCorrection){
                        localVolCol.setVisible(false);
                    }
                    if (!globalCorrection){
                        globalVolCol.setVisible(false);
                    }
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