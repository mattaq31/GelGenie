package qupath.ext.gelgenie.graphics;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Node;
import javafx.scene.SnapshotParameters;
import javafx.scene.chart.*;
import javafx.scene.image.WritableImage;
import qupath.fx.dialogs.FileChoosers;
import qupath.lib.common.GeneralTools;
import qupath.lib.images.servers.ServerTools;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;

import static qupath.lib.scripting.QP.getCurrentImageData;

/**
 * Main class in charge of embedded bar charts in main GelGenie window and results table.
 */
public class EmbeddedBarChart {
    public static ObservableList<XYChart.Series<String, Number>> plotBars(Collection<double[]> y_data,
                                                                          Collection<String> legendData,
                                                                          String[] labels) {
        // TODO: add normalisation option
        ObservableList<XYChart.Series<String, Number>> allPlots = FXCollections.observableArrayList();

        Iterator<double[]> itY = y_data.iterator();

        Iterator<String> itLegend = legendData.iterator();

        while (itY.hasNext()) {
            ObservableList<XYChart.Data<String, Number>> list = FXCollections.observableArrayList();
            double[] y = itY.next();
            String legend = itLegend.next();
            for (int i = 0; i < y.length; i++) {
                list.add(new XYChart.Data(labels[i], y[i]));
            }
            // create a series from the list
            XYChart.Series<String, Number> barSeries = new XYChart.Series<>(list);
            barSeries.setName(legend);
            allPlots.add(barSeries);
        }

        return allPlots;
    }

    /**
     * Given a set of pixel data, computes intensity histograms and returns them for display.
     *
     * @param y_data: Collection of arrays containing pixel data for individual annotations.
     * @param bins:   Number of bins to include in histogram.
     * @return Histogram series ready for plotting.
     */
    public static ObservableList<XYChart.Series<String, Number>> plotHistogram(Collection<double[]> y_data, int bins,
                                                                               Collection<String> labels){

        double max_val = 0.0;
        double min_val = 1000000;
        for (double[] y : y_data) { //computes combined min and max values from data
            max_val = Math.max(max_val, Arrays.stream(y).max().getAsDouble());
            min_val = Math.min(min_val, Arrays.stream(y).min().getAsDouble());
        }

        // Calculates the bin width from full range
        double binWidth = (max_val - min_val) / bins;

        ObservableList<XYChart.Series<String, Number>> allPlots = FXCollections.observableArrayList();

        Iterator<double[]> itY = y_data.iterator();
        Iterator<String> itLegend = labels.iterator();

        while (itY.hasNext()) {
            double[] y = itY.next();
            String legend = itLegend.next();

            ObservableList<XYChart.Data<String, Number>> list = FXCollections.observableArrayList();
            // Create bins and initialize their counts to 0
            int[] bin_values = new int[bins];
            for (double value : y) { // collects count data
                int binIndex = (int) ((value - min_val) / binWidth);
                if (binIndex < 0) {
                    binIndex = 0;
                } else if (binIndex >= bins) {
                    binIndex = bins - 1;
                }
                bin_values[binIndex]++;
            }

            // Add all values from the array into the list
            for (int i = 0; i < bin_values.length; i++) {
                double binStart = min_val + i * binWidth;
                double binEnd = min_val + (i + 1) * binWidth;
                list.add(new XYChart.Data(String.format("%.1f - %.1f", binStart, binEnd), bin_values[i]));
            }
            // create a series from the list
            XYChart.Series<String, Number> histoSeries = new XYChart.Series<>(list);
            histoSeries.setName(legend);
            allPlots.add(histoSeries);
        }

        return allPlots;
    }

    /**
     * Saves a bar chart to a file, prompting user to ask where it should be saved.
     * @param Chart: Chart to save
     */
    public static void saveChart(BarChart Chart){
        String defaultName = GeneralTools.getNameWithoutExtension(new File(ServerTools.getDisplayableImageName(getCurrentImageData().getServer())));

        File fileOutput = FileChoosers.promptToSaveFile("Export Chart",  new File(defaultName + "_band_chart.png"),
                FileChoosers.createExtensionFilter("Save as PNG", ".png"));
        if (fileOutput == null)
            return;

        SnapshotParameters snapshotParameters = new SnapshotParameters();
        snapshotParameters.setTransform(javafx.scene.transform.Scale.scale(4.0, 4.0)); // Increases the scale in an effort to obtain nicer images

        // to get the chart to look good with a white background, all text needs to be changed to black
        Chart.getXAxis().lookup(".axis-label").setStyle("-fx-text-fill: black;");
        Chart.getYAxis().lookup(".axis-label").setStyle("-fx-text-fill: black; -fx-tick-label-fill: black;");
        Chart.lookup(".chart-title").setStyle("-fx-text-fill: black;");
        Chart.getYAxis().setStyle("-fx-tick-label-fill: black;");
        Chart.getXAxis().setStyle("-fx-tick-label-fill: black;");
        for (Node node : Chart.lookupAll(".chart-legend-item")) {
            node.setStyle("-fx-text-fill: black;");
        }

        WritableImage image = Chart.snapshot(snapshotParameters, null);

        // all changed text needs to be reset to the defaults (to match system style) after writing out the snapshot
        Chart.getXAxis().lookup(".axis-label").setStyle(null);
        Chart.getYAxis().lookup(".axis-label").setStyle(null);
        Chart.lookup(".chart-title").setStyle(null);
        Chart.getYAxis().setStyle(null);
        Chart.getXAxis().setStyle(null);
        for (Node node : Chart.lookupAll(".chart-legend-item")) {
            node.setStyle(null);
        }

        try {
            ImageIO.write(SwingFXUtils.fromFXImage(image, null), "png", fileOutput);
        } catch (IOException e) { //TODO: add a proper error message
            e.printStackTrace();
        }

    }
}
