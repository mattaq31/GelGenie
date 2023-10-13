package qupath.ext.gelgenie.graphics;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.chart.*;

import java.util.Arrays;
import java.util.Collection;

public class EmbeddedBarChart {

    public ObservableList<XYChart.Series<String, Number>> plot(Collection<double[]> y_data, int bins) {
// TODO: documentation
        double max_val = 0.0;
        double min_val = 1000000;
        for (double[] y : y_data) {
            max_val = Math.max(max_val, Arrays.stream(y).max().getAsDouble());
            min_val = Math.min(min_val, Arrays.stream(y).min().getAsDouble());
        }

        // Calculate the bin width
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
            XYChart.Series<String, Number> stst = new XYChart.Series<>(list);
            stst.setName(String.format("Histogram %d", counter));
            allPlots.add(stst);
        }

        return allPlots;
    }

}
