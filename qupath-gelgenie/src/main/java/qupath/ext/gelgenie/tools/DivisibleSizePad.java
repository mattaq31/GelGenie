package qupath.ext.gelgenie.tools;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Transform;
import qupath.ext.gelgenie.models.PytorchManager;
import qupath.ext.gelgenie.ui.GelGeniePrefs;

public class DivisibleSizePad implements Transform {

    int multiplier;
    public DivisibleSizePad(int multiplier){
        this.multiplier = multiplier;
    }
    @Override
    public NDArray transform(NDArray array) {
        int currWidth = (int) array.getShape().getShape()[2];
        int currHeight = (int) array.getShape().getShape()[1];
        int padWidth = (int) (Math.ceil((double) currWidth / multiplier) * multiplier);
        int padHeight = (int) (Math.ceil((double) currHeight / multiplier) * multiplier);
        NDArray fin;
        try (NDManager manager = NDManager.newBaseManager(PytorchManager.getDevice())) {
            NDArray x = manager.zeros(new Shape(array.getShape().getShape()[0], currHeight, padWidth-currWidth));
            NDArray y = manager.zeros(new Shape(array.getShape().getShape()[0], padHeight-currHeight, padWidth));
            fin = array.concat(x, 2).concat(y, 1);
        }

        return fin;
    }
}
