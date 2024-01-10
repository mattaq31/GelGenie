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

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.CategoryMask;
import ai.djl.modality.cv.translator.BaseImageTranslator;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


/**
 * A {@link Translator} that post-process the {@link Image} into {@link CategoryMask} with output
 * mask representing the class that each pixel in the original image belong to.
 */
public class GelSegmentationTranslator extends BaseImageTranslator<CategoryMask> {

    private SynsetLoader synsetLoader;
    public List<String> classes;

    public int imageWidth;
    public int imageHeight;

    public GelSegmentationTranslator(Builder builder, int imageWidth, int imageHeight) {

        super(builder);
        this.synsetLoader = builder.synsetLoader();
        this.imageHeight = imageHeight;
        this.imageWidth = imageWidth;
    }

    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        if (classes == null) { // usually reads this from file, in this case we can just hardcode it once
            classes = new ArrayList<>();
            classes.add("Background");
            classes.add("Gel Band");
        }
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image image) {
        return super.processInput(ctx, image);
    }

    @Override
    public CategoryMask processOutput(TranslatorContext ctx, NDList list) {
        // scores contains the probabilities of each pixel being a certain object
        // important: the padding applied prior to running the model needs to be removed here
        float[] scores = list.get(0).get(":,0:" +imageHeight + ",0:" + imageWidth).toFloatArray();

        int[][] mask = new int[imageHeight][imageWidth];

        int imageSize = imageWidth * imageHeight;

        // Build mask array
        int numOfClasses = classes.size();
        for (int h = 0; h < imageHeight; h++) {
            for (int w = 0; w < imageWidth; w++) {
                int index = h * imageWidth + w;
                int maxi = 0;
                double maxnum = -Double.MAX_VALUE;
                for (int i = 0; i < numOfClasses; ++i) {
                    // get score for each i at the h,w pixel of the image
                    float score = scores[i * imageSize + index];
                    if (score > maxnum) {
                        maxnum = score;
                        maxi = i;
                    }
                }
                mask[h][w] = maxi;
            }
        }
        return new CategoryMask(classes, mask);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static Builder builder(Map<String, ?> arguments) {
        Builder builder = new Builder();

        builder.configPreProcess(arguments);
        builder.configPostProcess(arguments);

        return new Builder();
    }

    /**
     * This class is what's created first.  Once it's built (and additional transforms are appended), the .build() method
     * generates the actual translator class which is then passed on to the model criterion system.
     */
    public static class Builder extends ClassificationBuilder<Builder> {

        Builder() {}

        private SynsetLoader synsetLoader() {
            return super.synsetLoader;
        }

        @Override
        protected Builder self() {
            return this;
        }

        public GelSegmentationTranslator build(int imageWidth, int imageHeight) {
            validate();
            return new GelSegmentationTranslator(this, imageWidth, imageHeight);
        }

        public void configPreProcess(Map<String, ?> arguments) {
            super.configPreProcess(arguments);
        }
        public void configPostProcess(Map<String, ?> arguments) {
            super.configPostProcess(arguments);
        }
    }

}
