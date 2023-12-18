package qupath.ext.gelgenie.djl_processing;

import ai.djl.ndarray.NDArray;
import ai.djl.translate.Transform;


public class ChannelSquisher implements Transform {

    /**
     * DJL transform that squishes an RGB image into a grayscale image.  Only to be used when the RGB image
     * is also grayscale and not to actually convert between true RGB and grayscale.
     * @param array the {@link NDArray} on which the {@link Transform} is applied
     * @return all channels averaged into a single channel
     */
    @Override
    public NDArray transform(NDArray array) {

        return array.mean(new int[] {0}, true);
    }
}