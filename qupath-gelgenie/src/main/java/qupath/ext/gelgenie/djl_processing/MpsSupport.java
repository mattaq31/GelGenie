/**
 * Copyright 2023 University of Edinburgh
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
 *
 * Copied from qupath-extension-wsinfer
 */

package qupath.ext.gelgenie.djl_processing;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Transform;


/**
 * This class contains subclasses closely based upon classes in Deep Java Library (Apache v2.0,
 * see https://github.com/deepjavalibrary/djl)
 * <p>
 * They have been adapted as described below to work around issues with MPS devices on Apple Silicon.
 * <p>
 * The classes here should be removed if future updates to DJL make them unnecessary.
 * <p>
 * It should be possible to use these classes for other devices, since the changes do not restrict use to MPS.
 */
public class MpsSupport {

    /**
     * This is based upon the {@link ai.djl.modality.cv.transform.ToTensor} class, but adapted to work with MPS devices
     * on Apple Silicon.
     * <p>
     * The reason the original class can't be used is {@code result.div(255.0)} requires a float64, which fails on MPS.
     * The only required change is to use {@code result.div(255.0f)} instead.
     * <p>
     * If the original class is changed to use float32, then this class can be removed.
     */
    public static class ToTensor32 implements Transform {

        @Override
        public NDArray transform(NDArray array) {
            return toTensor32(array);
        }

        private NDArray toTensor32(NDArray array) {
            var manager = array.getNDArrayInternal().getArray().getManager();
            try (NDManager subManager = manager.newSubManager()) {
                array = array.getNDArrayInternal().getArray();
                array.attach(subManager);

                NDArray result = array;
                int dim = result.getShape().dimension();
                if (dim == 3) {
                    result = result.expandDims(0);
                }
                result = result.div(255.0f).transpose(0, 3, 1, 2);
                if (dim == 3) {
                    result = result.squeeze(0);
                }
                // The network by default takes float32
                if (!result.getDataType().equals(DataType.FLOAT32)) {
                    result = result.toType(DataType.FLOAT32, false);
                }
                array.attach(manager);
                result.attach(manager);
                return result;
            }
        }
    }
}
