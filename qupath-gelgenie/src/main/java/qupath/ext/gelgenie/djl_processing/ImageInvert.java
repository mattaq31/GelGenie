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


public class ImageInvert implements Transform {

    /**
     * DJL transform that inverts an image (i.e. 255 -> 0, 125 -> 130, similar for normalized values).
     * Should only be used to convert white backgrounds to black backgrounds,
     * as models only trained on black backgrounds.
     * @param array the {@link NDArray} on which the {@link Transform} is applied
     * @return same sized array, with inverted pixel values
     */
    @Override
    public NDArray transform(NDArray array) {
        NDArray ones = array.getManager().ones(array.getShape());
        return ones.sub(array);
    }
}