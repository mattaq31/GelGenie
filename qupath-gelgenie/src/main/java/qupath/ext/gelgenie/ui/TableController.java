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

import ij.ImagePlus;
import ij.plugin.filter.BackgroundSubtracter;

import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.collections.FXCollections;
import javafx.collections.ListChangeListener;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;

import javafx.geometry.Pos;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.gelgenie.graphics.EmbeddedBarChart;
import qupath.ext.gelgenie.tools.ImageTools;
import qupath.ext.gelgenie.tools.LaneBandCompare;
import qupath.fx.dialogs.Dialogs;
import qupath.imagej.tools.IJTools;
import qupath.lib.common.GeneralTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ServerTools;
import qupath.lib.objects.PathObject;

import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.*;
import java.util.concurrent.ExecutorService;

import qupath.fx.dialogs.FileChoosers;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.objects.hierarchy.events.PathObjectSelectionModel;
import qupath.lib.regions.RegionRequest;
import qupath.opencv.tools.OpenCVTools;

import static qupath.ext.gelgenie.graphics.EmbeddedBarChart.saveChart;
import static qupath.ext.gelgenie.tools.ImageTools.extractLocalBackgroundPixels;
import static qupath.imagej.images.servers.ImageJServer.convertToBufferedImage;
import static qupath.lib.scripting.QP.*;
import static qupath.lib.scripting.QP.getCurrentImageData;

public class TableController {

    //Table elements
    @FXML
    private TableView<BandEntry> mainTable;
    @FXML
    private TableColumn<BandEntry, PathObject> thumbnailCol;
    @FXML
    private TableColumn<BandEntry, Integer> bandCol;
    @FXML
    private TableColumn<BandEntry, Integer> laneCol;
    @FXML
    private TableColumn<BandEntry, String> nameCol;
    @FXML
    private TableColumn<BandEntry, Double> pixelCol;
    @FXML
    private TableColumn<BandEntry, Double> widthCol;
    @FXML
    private TableColumn<BandEntry, Double> heightCol;
    @FXML
    private TableColumn<BandEntry, Double> meanCol;
    @FXML
    private TableColumn<BandEntry, Double> stdCol;
    @FXML
    private TableColumn<BandEntry, Double> rawCol;
    @FXML
    private TableColumn<BandEntry, Double> localVolCol;
    @FXML
    private TableColumn<BandEntry, Double> globalVolCol;
    @FXML
    private TableColumn<BandEntry, Double> rollingVolCol;
    @FXML
    private TableColumn<BandEntry, Double> normVolCol;
    @FXML
    private TableColumn<BandEntry, Double> normLocalVolCol;
    @FXML
    private TableColumn<BandEntry, Double> normGlobalVolCol;
    @FXML
    private TableColumn<BandEntry, Double> normRollingVolCol;
    @FXML
    private SplitPane dataTableSplitPane;
    @FXML
    private Button histoButton;
    @FXML
    private Button exportButton;
    @FXML
    private Button exportHistoButton;
    @FXML
    private Button globalNormButton;
    @FXML
    private Button laneNormButton;
    private ObservableList<BandEntry> bandData;
    private Boolean barChartActive = false;

    private BorderPane histoPane;
    private CheckBox chartNormFlipper;
    private CheckBox chartRawView;
    private CheckBox chartGlobalView;
    private CheckBox chartLocalView;
    private CheckBox chartRollingView;
    private BarChart<String, Number> displayChart;

    public QuPathGUI qupath;
    private static final Logger logger = LoggerFactory.getLogger(TableController.class);

    // user-defined settings
    private boolean globalCorrection = false;
    private boolean localCorrection = false;
    private boolean rollingBallCorrection = false;
    private int localSensitivity = 5;
    private int rollingRadius = 50;
    private final ArrayList<PathObject> selectedBands = new ArrayList<>();
    private Mat rollingBallImage;

    double globalMean = 0.0;
    private final static ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.gelgenie.ui.strings");


    public TableController() {
        this.qupath = QuPathGUI.getInstance();
    }

    @FXML
    private void initialize() {
        // links table columns to properties of BandEntry class
        bandCol.setCellValueFactory(new PropertyValueFactory<>("bandID"));
        laneCol.setCellValueFactory(new PropertyValueFactory<>("laneID"));
        nameCol.setCellValueFactory(new PropertyValueFactory<>("bandName"));
        pixelCol.setCellValueFactory(new PropertyValueFactory<>("pixelCount"));
        widthCol.setCellValueFactory(new PropertyValueFactory<>("width"));
        heightCol.setCellValueFactory(new PropertyValueFactory<>("height"));
        meanCol.setCellValueFactory(new PropertyValueFactory<>("averageIntensity"));
        stdCol.setCellValueFactory(new PropertyValueFactory<>("stdIntensity"));
        rawCol.setCellValueFactory(new PropertyValueFactory<>("rawVolume"));
        localVolCol.setCellValueFactory(new PropertyValueFactory<>("localVolume"));
        globalVolCol.setCellValueFactory(new PropertyValueFactory<>("globalVolume"));
        rollingVolCol.setCellValueFactory(new PropertyValueFactory<>("rollingVolume"));
        normVolCol.setCellValueFactory(new PropertyValueFactory<>("normVolume"));
        normLocalVolCol.setCellValueFactory(new PropertyValueFactory<>("normLocal"));
        normGlobalVolCol.setCellValueFactory(new PropertyValueFactory<>("normGlobal"));
        normRollingVolCol.setCellValueFactory(new PropertyValueFactory<>("normRolling"));
        thumbnailCol.setCellValueFactory(new PropertyValueFactory<>("parentAnnotation"));

        // formatting for double columns
        widthCol.setCellFactory(TableController::getTableFormattedDouble);
        heightCol.setCellFactory(TableController::getTableFormattedDouble);
        meanCol.setCellFactory(TableController::getTableFormattedDouble);
        stdCol.setCellFactory(TableController::getTableFormattedDouble);
        localVolCol.setCellFactory(TableController::getTableFormattedDouble);
        globalVolCol.setCellFactory(TableController::getTableFormattedDouble);
        rollingVolCol.setCellFactory(TableController::getTableFormattedDouble);
        normVolCol.setCellFactory(TableController::getTableFormattedHighPrecision);
        normLocalVolCol.setCellFactory(TableController::getTableFormattedHighPrecision);
        normGlobalVolCol.setCellFactory(TableController::getTableFormattedHighPrecision);
        normRollingVolCol.setCellFactory(TableController::getTableFormattedHighPrecision);

        // getting data from viewer
        ImageData<BufferedImage> imageData = getCurrentImageData();
        ImageServer<BufferedImage> server = imageData.getServer();
        final PathObjectHierarchy hierarchy = imageData.getHierarchy();
        var viewer = qupath.getAllViewers().stream().filter(v -> v.getImageData() == imageData).findFirst().orElse(null);

        // uses the implementation in qupath to extract a thumbnail from an annotation
        thumbnailCol.setCellFactory(column -> createTableCellByReflection(
                viewer, imageData.getServer(), true, 5,
                qupath.getThreadPoolManager().getSingleThreadExecutor(this)));

        // Set fixed cell size for the thumbnail - this can avoid large numbers of non-visible cells being computed
        mainTable.fixedCellSizeProperty().bind(Bindings.createDoubleBinding(() -> {
            if (thumbnailCol.isVisible())
                return Math.max(24, thumbnailCol.getWidth() + 5);
            else
                return -1.0;
        }, thumbnailCol.widthProperty(), thumbnailCol.visibleProperty()));

        ArrayList<PathObject> annots = (ArrayList<PathObject>) getAnnotationObjects();

        // permanent chart settings
        final CategoryAxis xAxis = new CategoryAxis();
        final NumberAxis yAxis = new NumberAxis();
        displayChart = new BarChart<>(xAxis, yAxis);
        displayChart.setTitle("Visual Data Depiction");
        displayChart.lookup(".chart-plot-background").setStyle("-fx-background-color: transparent;");
        displayChart.lookup(".chart-legend").setStyle("-fx-background-color: transparent;");
        displayChart.setAnimated(false);
        xAxis.setLabel("Band");
        yAxis.setLabel("Intensity (A.U.)");

        // permanent table settings
        mainTable.setPlaceholder(new Label("No gel band data to display"));
        TableView.TableViewSelectionModel<BandEntry> selectionModel = mainTable.getSelectionModel();
        selectionModel.setSelectionMode(SelectionMode.MULTIPLE);
        ButtonBar.setButtonData(histoButton, ButtonBar.ButtonData.RIGHT);
        ButtonBar.setButtonData(exportButton, ButtonBar.ButtonData.RIGHT);
        ButtonBar.setButtonData(exportHistoButton, ButtonBar.ButtonData.RIGHT);
        ButtonBar.setButtonData(laneNormButton, ButtonBar.ButtonData.LEFT);
        ButtonBar.setButtonData(globalNormButton, ButtonBar.ButtonData.LEFT);

        globalNormButton.setDisable(true);

        // when the table is double-clicked, the viewer zooms in on the selected band
        mainTable.setRowFactory(params -> {
            var row = new TableRow<BandEntry>();
            row.setOnMouseClicked(e -> {
                if (e.getClickCount() == 2) {
                    if (row.getItem() == null)
                        return;
                    var roi = row.getItem().getParentAnnotation().getROI();
                    if (roi != null && viewer != null && viewer.getHierarchy() != null)
                        viewer.centerROI(roi);
                }
            });
            return row;
        });

        // the below synchronises selections on the table with selections in QuPath (and the histogram viewer)
        // However, the opposite is currently not implemented (would require more code from SummaryMeasurementTableCommand).
        mainTable.getSelectionModel().getSelectedItems().addListener((ListChangeListener<BandEntry>) c -> synchroniseTableSelection(hierarchy, c, mainTable));

        // Core table generation functionality.
        // This code block depends on user settings, which are not provided until this runLater() command.
        Platform.runLater(() -> {
            calculateGlobalBackground(server);
            try {
                rollingBallImage = findRollingBallImage(server, rollingRadius);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            if (!selectedBands.isEmpty()) {
                selectedBands.sort(new LaneBandCompare());
                bandData = computeTableColumns(selectedBands, server, globalCorrection, localCorrection, rollingBallCorrection,
                        localSensitivity, globalMean, rollingBallImage, "Global");
            } else {
                annots.sort(new LaneBandCompare());
                bandData = computeTableColumns(annots, server, globalCorrection, localCorrection, rollingBallCorrection,
                        localSensitivity, globalMean, rollingBallImage, "Global");
            }
            tableSetup();

            // chart frame setup and button config
            histoPane = new BorderPane();
            histoPane.setCenter(displayChart);
            chartNormFlipper = new CheckBox(resources.getString("ui.table.histo.norm"));
            chartNormFlipper.selectedProperty().addListener((observable, oldValue, newValue) -> updateHistogramData()); // chart data always updated when checkbox is toggled

            chartRawView = new CheckBox(resources.getString("ui.table.histo.raw"));
            chartRawView.selectedProperty().addListener((observable, oldValue, newValue) -> updateHistogramData()); // chart data always updated when checkbox is toggled

            chartGlobalView = new CheckBox(resources.getString("ui.table.histo.global"));
            chartGlobalView.selectedProperty().addListener((observable, oldValue, newValue) -> updateHistogramData()); // chart data always updated when checkbox is toggled

            chartRollingView = new CheckBox(resources.getString("ui.table.histo.rolling"));
            chartRollingView.selectedProperty().addListener((observable, oldValue, newValue) -> updateHistogramData()); // chart data always updated when checkbox is toggled

            chartLocalView = new CheckBox(resources.getString("ui.table.histo.local"));
            chartLocalView.selectedProperty().addListener((observable, oldValue, newValue) -> updateHistogramData()); // chart data always updated when checkbox is toggled

            HBox chartFooter = new HBox(chartNormFlipper, chartRawView);

            if (globalCorrection) {
                chartFooter.getChildren().add(chartGlobalView);
                HBox.setMargin(chartGlobalView, new javafx.geometry.Insets(0, 5, 10, 5));

            }
            if (localCorrection) {
                chartFooter.getChildren().add(chartLocalView);
                HBox.setMargin(chartLocalView, new javafx.geometry.Insets(0, 5, 10, 5));

            }
            if (rollingBallCorrection) {
                chartFooter.getChildren().add(chartRollingView);
                HBox.setMargin(chartRollingView, new javafx.geometry.Insets(0, 0, 10, 5));
            }

            chartRawView.setSelected(true);
            chartFooter.setAlignment(Pos.CENTER);
            histoPane.setBottom(chartFooter);
            HBox.setMargin(chartNormFlipper, new javafx.geometry.Insets(0, 5, 10, 0));
            HBox.setMargin(chartRawView, new javafx.geometry.Insets(0, 5, 10, 5));

            // toggleHistogram(); Having the histogram pop up by default can be annoying sometimes.
        });
    }

    /**
     * Formats double columns to two decimal places.
     *
     * @param bandEntryDoubleTableColumn Column being edited.
     * @return Edited value.
     */
    private static TableCell<BandEntry, Double> getTableFormattedDouble(TableColumn<BandEntry, Double> bandEntryDoubleTableColumn) {
        return new TableCell<>() {
            @Override
            protected void updateItem(Double value, boolean empty) {
                super.updateItem(value, empty);
                if (empty) {
                    setText(null);
                } else {
                    setText(String.format("%.2f", value.floatValue()));
                }
            }
        };
    }

    private static TableCell<BandEntry, Double> getTableFormattedHighPrecision(TableColumn<BandEntry, Double> bandEntryDoubleTableColumn) {
        return new TableCell<>() {
            @Override
            protected void updateItem(Double value, boolean empty) {
                super.updateItem(value, empty);
                if (empty) {
                    setText(null);
                } else {
                    setText(String.format("%.3f", value.floatValue()));
                }
            }
        };
    }

    /**
     * Pre-computes global background value on user-selected patch.
     *
     * @param server: Server corresponding to open image.
     */
    private void calculateGlobalBackground(ImageServer<BufferedImage> server) {
        if (globalCorrection) {
            try {
                globalMean = calculateGlobalBackgroundAverage(server);
            } catch (Exception e) {
                globalCorrection = false;
                Dialogs.showInfoNotification(
                        resources.getString("title"),
                        resources.getString("ui.processing.no-global-mean"));
            }
        }
    }
    /**
     * Computes the average pixel value of global background patches.
     *
     * @param server: Server corresponding to current image
     * @return Computed global mean.
     * @throws Exception
     */
    private static double calculateGlobalBackgroundAverage(ImageServer<BufferedImage> server) throws Exception {
        double global_mean = 0.0;
        Collection<PathObject> annots = getAnnotationObjects();
        for (PathObject annot : annots) {
            if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Global Background")) {
                double[] all_pixels = ImageTools.extractAnnotationPixels(annot, server);
                global_mean = global_mean + Arrays.stream(all_pixels).average().getAsDouble();
            }
        }
        if (global_mean == 0.0) {
            throw new Exception("No annotation marked for global background correction");
        }
        return global_mean;
    }

    /**
     * Passes the rolling ball filter over the entire input image and stores it for downstream processing.
     *
     * @param server: Main image server
     * @throws IOException
     * @return: Rolling ball filtered image
     */
    private static Mat findRollingBallImage(ImageServer<BufferedImage> server, int rollingRadius) throws IOException {

        RegionRequest request = RegionRequest.createInstance(server, 1.0); // generates full image request

        ImagePlus imp = IJTools.convertToImagePlus(server, request).getImage(); // converts to ImageJ format

        BackgroundSubtracter bs = new BackgroundSubtracter(); // creates background subtracter

        // all default settings used except for rollingRadius, which can be user-defined
        bs.rollingBallBackground(imp.getProcessor(), rollingRadius, false, false, false, false, false);

        // converts to OpenCV image for downstream processing
        BufferedImage subtractedImage = convertToBufferedImage(imp, 0, 0, null);
        return OpenCVTools.imageToMat(subtractedImage);
    }

    /**
     * Computes all measurements for each band in current image.
     *
     * @param annots: List of annotations in image.
     * @param server: Server corresponding to open image.
     */
    private static ObservableList<BandEntry> computeTableColumns(Collection<PathObject> annots, ImageServer<BufferedImage> server,
                                                                 boolean globalCorrection, boolean localCorrection, boolean rollingBallCorrection,
                                                                 int localSensitivity, double globalMean, Mat rollingBallImage, String normType) {

        ObservableList<BandEntry> all_bands = FXCollections.observableArrayList();

        for (PathObject annot : annots) {
            //  only act on annotations marked as bands
            if (annot.getPathClass() != null && Objects.equals(annot.getPathClass().getName(), "Gel Band")) {
                double[] all_pixels = ImageTools.extractAnnotationPixels(annot, server); // extracts a list of pixels matching the specific selected annotation

                // computes intensity average
                double pixel_average = Arrays.stream(all_pixels).average().getAsDouble();
                // Calculate the sum of squared differences
                double sumOfSquaredDifferences = Arrays.stream(all_pixels).map(num -> Math.pow(num - pixel_average, 2)).sum();
                // Calculate the standard deviation
                double pixel_std = Math.sqrt(sumOfSquaredDifferences / all_pixels.length);

                // raw band volume is simply a sum of all pixels
                double raw_volume = Arrays.stream(all_pixels).sum();

                // todo: does it make sense to mask all bands rather than just the selected one?
                double[] localBackgroundPixels = extractLocalBackgroundPixels(annot, server, localSensitivity);
                double localMean = Arrays.stream(localBackgroundPixels).average().getAsDouble();

                double width = annot.getROI().getBoundsWidth();
                double height = annot.getROI().getBoundsHeight();

                double globalVolume = 0;
                double localVolume = 0;
                double rollingBallVolume = 0;

                if (globalCorrection) {
                    globalVolume = raw_volume - (globalMean * all_pixels.length);
                }
                if (localCorrection) {
                    localVolume = raw_volume - (localMean * all_pixels.length);
                }
                if (rollingBallCorrection) { // rolling ball values need to be computed on its specific image
                    double[] rbPixels = ImageTools.extractAnnotationPixelsFromMat(annot, rollingBallImage);
                    rollingBallVolume = Arrays.stream(rbPixels).sum();
                }

                int bandID;
                int laneID;
                if (annot.getMeasurements().get("BandID") == null) {
                    bandID = 0;
                } else {
                    bandID = annot.getMeasurements().get("BandID").intValue();
                }

                if (annot.getMeasurements().get("LaneID") == null) {
                    laneID = 0;
                } else {
                    laneID = annot.getMeasurements().get("LaneID").intValue();
                }

                BandEntry curr_band = new BandEntry(bandID, laneID, annot.getName(), all_pixels.length,
                        width, height, pixel_average, pixel_std, raw_volume, globalVolume, localVolume,
                        rollingBallVolume, annot);
                all_bands.add(curr_band);
            }
        }
        if (normType.equals("Global")){
            fullNormalise(all_bands); // normalise globally (for all bands in image)
        }
        else if (normType.equals("Lane")){
            laneNormalise(all_bands); // normalise by lane
        }
        else{
            // some form of spelling mistake by user
            Dialogs.showWarningNotification(resources.getString("title"), resources.getString("error.wrong-norm"));
            return FXCollections.observableArrayList();
        }

        return all_bands;
    }

    public void laneNormalise(){
        laneNormalise(bandData);
        mainTable.refresh();
        globalNormButton.setDisable(false);
        laneNormButton.setDisable(true);
        updateHistogramData();
    }
    public void fullNormalise() {
        fullNormalise(bandData);
        mainTable.refresh();
        globalNormButton.setDisable(true);
        laneNormButton.setDisable(false);
        updateHistogramData();
    }
    /**
     * Normalises all columns to the minimum and maximum values in each lane (lane normalisation).
     * @param bands: List of bands to be normalised.
     */
    private static void laneNormalise(ObservableList<BandEntry> bands){
        double inf = Double.POSITIVE_INFINITY;
        Map<Integer, Double[][]> laneDictionary = new HashMap<>();
        List<Double> rawList = new ArrayList<>();

        for (BandEntry entry : bands) {
            if (!laneDictionary.containsKey(entry.getLaneID())){
                laneDictionary.put(entry.getLaneID(), new Double[][]{{inf, -inf}, {inf, -inf}, {inf, -inf}, {inf, -inf}});
            }
            rawList.add(entry.getRawVolume());
            rawList.add(entry.getGlobalVolume());
            rawList.add(entry.getLocalVolume());
            rawList.add(entry.getRollingVolume());
            int position = 0;
            for (double value: rawList){
                laneDictionary.get(entry.getLaneID())[position][0] = Math.min(laneDictionary.get(entry.getLaneID())[position][0], value);
                laneDictionary.get(entry.getLaneID())[position][1] = Math.max(laneDictionary.get(entry.getLaneID())[position][1], value);
                position++;
            }
            rawList.clear();
        }
        for (BandEntry entry : bands) {
            // this case happens when only one band is available in a lane - just set everything to 1.0
            if(Objects.equals(laneDictionary.get(entry.getLaneID())[0][0], laneDictionary.get(entry.getLaneID())[0][1])){
                entry.setNormVolume(1.0);
                entry.setNormGlobal(1.0);
                entry.setNormLocal(1.0);
                entry.setNormRolling(1.0);
            }
            else {
                entry.setNormVolume((entry.getRawVolume() - laneDictionary.get(entry.getLaneID())[0][0]) / (laneDictionary.get(entry.getLaneID())[0][1] - laneDictionary.get(entry.getLaneID())[0][0]));
                entry.setNormGlobal((entry.getGlobalVolume() - laneDictionary.get(entry.getLaneID())[1][0]) / (laneDictionary.get(entry.getLaneID())[1][1] - laneDictionary.get(entry.getLaneID())[1][0]));
                entry.setNormLocal((entry.getLocalVolume() - laneDictionary.get(entry.getLaneID())[2][0]) / (laneDictionary.get(entry.getLaneID())[2][1] - laneDictionary.get(entry.getLaneID())[2][0]));
                entry.setNormRolling((entry.getRollingVolume() - laneDictionary.get(entry.getLaneID())[3][0]) / (laneDictionary.get(entry.getLaneID())[3][1] - laneDictionary.get(entry.getLaneID())[3][0]));
            }
        }
    }

    /**
     * Normalises all columns to the minimum and maximum values in the table (global normalisation).
     * @param bands: List of bands to be normalised.
     */
    private static void fullNormalise(ObservableList<BandEntry> bands){
        double inf = Double.POSITIVE_INFINITY;
        double[] minMax = {inf, -inf};
        double[] globalMinMax = {inf, -inf};
        double[] localMinMax = {inf, -inf};
        double[] rollingMinMax = {inf, -inf};

        for (BandEntry entry : bands) {
            minMax[0] = Math.min(minMax[0], entry.getRawVolume());
            minMax[1] = Math.max(minMax[1], entry.getRawVolume());
            localMinMax[0] = Math.min(localMinMax[0], entry.getLocalVolume());
            localMinMax[1] = Math.max(localMinMax[1], entry.getLocalVolume());
            globalMinMax[0] = Math.min(globalMinMax[0], entry.getGlobalVolume());
            globalMinMax[1] = Math.max(globalMinMax[1], entry.getGlobalVolume());
            rollingMinMax[0] = Math.min(rollingMinMax[0], entry.getRollingVolume());
            rollingMinMax[1] = Math.max(rollingMinMax[1], entry.getRollingVolume());
        }

        for (BandEntry entry : bands) {
            entry.setNormVolume((entry.getRawVolume() - minMax[0]) / (minMax[1] - minMax[0]));
            entry.setNormLocal((entry.getLocalVolume() - localMinMax[0]) / (localMinMax[1] - localMinMax[0]));
            entry.setNormGlobal((entry.getGlobalVolume() - globalMinMax[0]) / (globalMinMax[1] - globalMinMax[0]));
            entry.setNormRolling((entry.getRollingVolume() - rollingMinMax[0]) / (rollingMinMax[1] - rollingMinMax[0]));
        }
    }

    /**
     * Adds all data to visible table and hides columns based on user preferences.
     */
    private void tableSetup() {
        mainTable.setItems(bandData);
        if (!localCorrection) {
            localVolCol.setVisible(false);
            normLocalVolCol.setVisible(false);
        }
        if (!globalCorrection) {
            globalVolCol.setVisible(false);
            normGlobalVolCol.setVisible(false);
        }
        if (!rollingBallCorrection) {
            rollingVolCol.setVisible(false);
            normRollingVolCol.setVisible(false);
        }
    }

    /**
     * Exports salient table columns to CSV.
     * @param bandData: List of bandEntry datapoints
     * @param folder: Folder to save data to with default or specified filename
     * @throws IOException
     */
    public static void exportDataToFolder(ObservableList<BandEntry> bandData, String folder, String filename,
                                          boolean globalCorrection, boolean localCorrection,
                                          boolean rollingBallCorrection) throws IOException {
        String defaultName = GeneralTools.getNameWithoutExtension(new File(ServerTools.getDisplayableImageName(getCurrentImageData().getServer())));
        File fileOutput;
        if (filename == null){
            fileOutput = new File(folder + "/" + defaultName + "_band_data.csv");
        }
        else {
            fileOutput = new File(folder + "/" + filename);
        }
        exportData(bandData, fileOutput, globalCorrection, localCorrection, rollingBallCorrection);
    }

    /**
     * Exports salient table columns to CSV.
     * @param bandData: List of bandEntry datapoints
     * @param filename: Filename to save data to
     * @throws IOException
     */
    public static void exportData(ObservableList<BandEntry> bandData, File filename, boolean globalCorrection, boolean localCorrection, boolean rollingBallCorrection) throws IOException {

        BufferedWriter br = new BufferedWriter(new FileWriter(filename));
        String headerString = "Name,Lane ID,Band ID,Pixel Count,Width,Height,Average Intensity,Intensity SD,Raw Volume,Norm. Raw Volume";
        if (localCorrection) {
            headerString = headerString + ",Local Corrected Volume,Norm. Local Volume";
        }
        if (globalCorrection) {
            headerString = headerString + ",Global Corrected Volume,Norm. Global Volume";
        }
        if (rollingBallCorrection) {
            headerString = headerString + ",Rolling Ball Corrected Volume,Norm. Rolling Volume";
        }
        headerString = headerString + "\n";
        br.write(headerString);

        for (BandEntry band : bandData) {
            String sb = band.getBandName() + "," + band.getLaneID() + "," + band.getBandID() + "," + band.getPixelCount() + "," + band.getWidth()
                    + "," + band.getHeight() + "," + band.getAverageIntensity() + "," + band.getStdIntensity()
                    + "," + band.getRawVolume() + "," + band.getNormVolume();
            if (localCorrection) {
                sb = sb + "," + band.getLocalVolume() + "," + band.getNormLocal();
            }
            if (globalCorrection) {
                sb = sb + "," + band.getGlobalVolume() + "," + band.getNormGlobal();
            }
            if (rollingBallCorrection) {
                sb = sb + "," + band.getRollingVolume() + "," + band.getNormRolling();
            }
            sb = sb + "\n";
            br.write(sb);
        }
        br.close();
    }
    /**
     * Exports salient table columns to CSV (this is the function called by the UI button that uses the global saved value of the bandData).
     *
     * @throws IOException
     */
    public void exportData() throws IOException {

        String defaultName = GeneralTools.getNameWithoutExtension(new File(ServerTools.getDisplayableImageName(getCurrentImageData().getServer())));

        File fileOutput = FileChoosers.promptToSaveFile("Export image region", new File(defaultName + "_band_data.csv"),
                FileChoosers.createExtensionFilter("Set CSV output filename", ".csv"));
        if (fileOutput == null)
            return;
        exportData(bandData, fileOutput, globalCorrection, localCorrection, rollingBallCorrection);
    }

    /**
     * Toggles the display of a histogram, which appears as a side panel in the table view.
     */
    public void toggleHistogram() {
        if (barChartActive) {
            dataTableSplitPane.getItems().remove(1);
            barChartActive = false;
        } else {
            updateHistogramData();
            dataTableSplitPane.getItems().add(histoPane);
            barChartActive = true;
        }
    }

    /**
     * Collects data from table and prepares histogram.  Currently only presents raw and global corrected volumes.
     */
    private void updateHistogramData() {

        boolean rawIsValid = chartRawView.isSelected();
        boolean globalIsValid = this.globalCorrection && chartGlobalView.isSelected();
        boolean localIsValid = this.localCorrection && chartLocalView.isSelected();
        boolean rollingIsValid = this.rollingBallCorrection && chartRollingView.isSelected();

        if (chartNormFlipper.isSelected()){
            displayChart.getYAxis().setLabel("Normalised Intensity");
        }else{
            displayChart.getYAxis().setLabel("Intensity (A.U.)");
        }

        Collection<double[]> dataList = new ArrayList<>();
        Collection<String> legendList = new ArrayList<>();

        ObservableList<BandEntry> all_bands = mainTable.getItems();
        double[] rawPixels = new double[all_bands.size()];
        double[] globalCorrVol = new double[all_bands.size()];
        double[] localCorrVol = new double[all_bands.size()];
        double[] rollingCorrVol = new double[all_bands.size()];

        String[] labels = new String[all_bands.size()];
        int counter = 0;

        for (BandEntry band : all_bands) {
            labels[counter] = band.getBandName();

            if (chartNormFlipper.isSelected()){
                if (rawIsValid) {
                    rawPixels[counter] = band.getNormVolume();
                }
                if (globalIsValid) {
                    globalCorrVol[counter] = band.getNormGlobal();
                }
                if (localIsValid) {
                    localCorrVol[counter] = band.getNormLocal();
                }
                if (rollingIsValid) {
                    rollingCorrVol[counter] = band.getNormRolling();
                }
            }
            else {
                if (rawIsValid) {
                    rawPixels[counter] = band.getRawVolume();
                }
                if (globalIsValid) {
                    globalCorrVol[counter] = band.getGlobalVolume();
                }
                if (localIsValid) {
                    localCorrVol[counter] = band.getLocalVolume();
                }
                if (rollingIsValid) {
                    rollingCorrVol[counter] = band.getRollingVolume();
                }
            }
            counter++;
        }

        if (rawIsValid) {
            dataList.add(rawPixels);
            legendList.add("Raw Volume");
        }

        if (globalIsValid) {
            dataList.add(globalCorrVol);
            legendList.add("Global Corrected Volume");
        }

        if (localIsValid) {
            dataList.add(localCorrVol);
            legendList.add("Local Corrected Volume");
        }
        if (rollingIsValid) {
            dataList.add(rollingCorrVol);
            legendList.add("Rolling Ball Corrected Volume");
        }

        ObservableList<XYChart.Series<String, Number>> allPlots = EmbeddedBarChart.plotBars(dataList, legendList, labels);

        displayChart.getData().clear(); // removes previous data
        displayChart.getData().addAll(allPlots);
    }

    /**
     * Saves histogram to file.
     */
    public void saveHistogram() {
        saveChart(displayChart);
    }

    /**
     * Sets user preferences.
     *
     * @param globalCorrection:      Enable global background calculation
     * @param localCorrection:       Enable local background calculation
     * @param rollingBallCorrection: Enable rolling ball background calculation
     * @param localSensitivity:      Pixel sensitivity for local background calculation
     * @param rollingRadius:         The radius used for the rolling ball background subtraction
     * @param selectedBands:         List of selected bands on which to compute data
     */
    public void setPreferences(boolean globalCorrection, boolean localCorrection, boolean rollingBallCorrection,
                               int localSensitivity, int rollingRadius, Collection<PathObject> selectedBands) {
        this.localCorrection = localCorrection;
        this.globalCorrection = globalCorrection;
        this.rollingBallCorrection = rollingBallCorrection;
        this.localSensitivity = localSensitivity;
        this.rollingRadius = rollingRadius;

        if (!selectedBands.isEmpty()) {
            this.selectedBands.addAll(selectedBands);
        }
    }

    /**
     * Scriptable function to compute and export band data without producing an explicit table window.
     * @param globalCorrection:      Enable global background calculation
     * @param localCorrection:       Enable local background calculation
     * @param rollingBallCorrection: Enable rolling ball background calculation
     * @param localSensitivity:      Pixel sensitivity for local background calculation
     * @param rollingRadius:         The radius used for the rolling ball background subtraction
     * @param folder: Folder to save data to
     * @param filename: Specific filename to use or null to use default filename
     * @throws Exception
     */
    public static void computeAndExportBandData(boolean globalCorrection, boolean localCorrection, boolean rollingBallCorrection,
                                                String normType,
                                                int localSensitivity, int rollingRadius, String folder, String filename) throws Exception {

        ImageServer<BufferedImage> server = getCurrentImageData().getServer();
        double globalMean = calculateGlobalBackgroundAverage(server);
        Mat rollingBallImage = findRollingBallImage(server, rollingRadius);

        ArrayList<PathObject> annots = (ArrayList<PathObject>) getAnnotationObjects(); // sorts annotations by lane/band ID since this sorting is lost after reloading an image
        annots.sort(new LaneBandCompare());

        ObservableList<BandEntry> bandData = computeTableColumns(annots, server, globalCorrection, localCorrection, rollingBallCorrection, localSensitivity, globalMean, rollingBallImage, normType);
        exportDataToFolder(bandData, folder, filename, globalCorrection, localCorrection, rollingBallCorrection);
    }

    /**
     * Provides access to QuPath's tablecell thumbnail creation method via reflection (provided by Pete).
     *
     * @param viewer:      Current image viewer
     * @param server:      Server corresponding to current image
     * @param paintObject: Set to true to paint out the annotation within the thumbnail
     * @param padding:     Padding to include around image
     * @param pool:        Thread pool
     * @return Updated table cell with thumbnail
     */
    public static <S extends PathObject, T extends PathObject> TableCell<S, T> createTableCellByReflection(
            QuPathViewer viewer, ImageServer<BufferedImage> server, boolean paintObject, double padding, ExecutorService pool
    ) {
        Class<?> cls = null;
        try {
            cls = Class.forName("qupath.lib.gui.commands.PathObjectImageManagers");
            var method = cls.getDeclaredMethod("createTableCell",
                    QuPathViewer.class,
                    ImageServer.class,
                    boolean.class,
                    double.class,
                    ExecutorService.class
            );
            method.setAccessible(true);
            return (TableCell) method.invoke(null, viewer, server, paintObject, padding, pool);
        } catch (ClassNotFoundException | InvocationTargetException | NoSuchMethodException |
                 IllegalAccessException e) {
            logger.warn("Exception creating table cell: {}", e.getMessage(), e);
            return new TableCell<>();
        }
    }

    /**
     * This function synchronises a selection in the table with the selection in the rest of QuPath.
     * This allows users to view their selected band's pixel distribution, even from within a table.
     */
    private void synchroniseTableSelection(final PathObjectHierarchy hierarchy,
                                           final ListChangeListener.Change<? extends BandEntry> change,
                                           final TableView<BandEntry> table) {
        if (hierarchy == null) return;

        PathObjectSelectionModel model = hierarchy.getSelectionModel();
        if (model == null) return;

        // Checks if anything was removed
        boolean removed = false;
        if (change != null) {
            while (change.next())
                removed = removed | change.wasRemoved();
        }

        MultipleSelectionModel<BandEntry> treeModel = table.getSelectionModel();
        List<BandEntry> selectedItems = treeModel.getSelectedItems();

        // If we just have no selected items, and something was removed, then clear the selection
        if (selectedItems.isEmpty() && removed) {
            model.clearSelection();
            return;
        }

        // If there is just one selected item, and also items were removed from the selection, then only select the one item we have
        if (selectedItems.size() == 1) {
            model.setSelectedObject(selectedItems.get(0).getParentAnnotation(), false);
            return;
        }

        // If there are multiple selected items, need to ensure that everything in the tree matches with everything in the selection model
        Set<BandEntry> toSelect = new HashSet<>(treeModel.getSelectedItems());
        Collection<PathObject> annotSelect = new ArrayList<PathObject>();

        for (BandEntry entry : toSelect) {
            annotSelect.add(entry.getParentAnnotation());
        }

        PathObject primary = treeModel.getSelectedItem().getParentAnnotation();
        model.setSelectedObjects(annotSelect, primary);
    }

}