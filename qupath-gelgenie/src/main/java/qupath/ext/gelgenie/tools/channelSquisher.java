package qupath.ext.gelgenie.tools;

import ai.djl.ndarray.NDArray;
import ai.djl.translate.Transform;

public class channelSquisher implements Transform {

    @Override
    public NDArray transform(NDArray array) {

        return array.mean(new int[] {0}, true);
    }
}