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

package qupath.ext.gelgenie.djl_processing;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Transform;

public class DivisibleSizePad implements Transform {

    int leftPad;
    int rightPad;
    int topPad;
    int bottomPad;

    public DivisibleSizePad(int leftPad, int rightPad, int topPad, int bottomPad){
        this.leftPad = leftPad;
        this.rightPad = rightPad;
        this.topPad = topPad;
        this.bottomPad = bottomPad;
    }

    /**
     * Pads an image with zeros to match some specific multiplier (e.g. UNet needs images to be divisible by 32)
     * @param array the {@link NDArray} on which the {@link Transform} is applied
     * @return The image with padding applied
     */
    @Override
    public NDArray transform(NDArray array) {
        int currWidth = (int) array.getShape().getShape()[2];
        int currHeight = (int) array.getShape().getShape()[1];

        NDArray xLeft = array.getManager().zeros(new Shape(array.getShape().getShape()[0], currHeight, leftPad));
        NDArray xRight = array.getManager().zeros(new Shape(array.getShape().getShape()[0], currHeight, rightPad));
        NDArray yTop = array.getManager().zeros(new Shape(array.getShape().getShape()[0], topPad, currWidth + leftPad + rightPad));
        NDArray yBottom = array.getManager().zeros(new Shape(array.getShape().getShape()[0], bottomPad, currWidth + leftPad + rightPad));

        return yTop.concat(xLeft.concat(array, 2).concat(xRight, 2), 1).concat(yBottom, 1); // padding effect is achieved by combining the original array and all the zero arrays
    }
}
