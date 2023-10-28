package qupath.ext.gelgenie.graphics;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.chart.*;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;

/**
 * Main class in charge of embedded bar chart.
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
    public static ObservableList<XYChart.Series<String, Number>> plotHistogram(Collection<double[]> y_data, int bins) {

        double max_val = 0.0;
        double min_val = 1000000;
        for (double[] y : y_data) { //computes combined min and max values from data
            max_val = Math.max(max_val, Arrays.stream(y).max().getAsDouble());
            min_val = Math.min(min_val, Arrays.stream(y).min().getAsDouble());
        }

        // Calculates the bin width from full range
        double binWidth = (max_val - min_val) / bins;

        ObservableList<XYChart.Series<String, Number>> allPlots = FXCollections.observableArrayList();
        int counter = 0;

        for (double[] y : y_data) {
            ObservableList<XYChart.Data<String, Number>> list = FXCollections.observableArrayList();
            counter++;

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
            histoSeries.setName(String.format("Histogram %d", counter));
            allPlots.add(histoSeries);
        }

        return allPlots;
    }

}
