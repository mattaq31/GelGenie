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

package qupath.ext.gelgenie.tools;

import qupath.lib.common.ColorTools;
import qupath.lib.common.GeneralTools;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.LabeledImageServer;
import qupath.lib.scripting.QP;

import java.awt.image.BufferedImage;
import java.io.IOException;

import static qupath.lib.scripting.QP.*;

public class SegmentationMap {

    /**
     * Scripting friendly function for automatically generating filename for exporting a segmentation map
     * @throws IOException
     */
    public static void exportSegmentationMapToProjectFolder() throws IOException {
        // Define output path (relative to project)
        String outputDir = buildFilePath(PROJECT_BASE_DIR, "SegMaps");
        mkdirs(outputDir);
        String name = GeneralTools.stripExtension(QP.getCurrentImageData().getServer().getMetadata().getName());
        String path = buildFilePath(outputDir, name + "_SegMap.tif");
        exportSegmentationMap(path);
    }

    /**
     * Exports current segmentation map to file (white background, TIFF)
     * @param filename: full path to file
     * @throws IOException
     */
    public static void exportSegmentationMap(String filename) throws IOException {

        ImageData<BufferedImage> imageData = QP.getCurrentImageData();
        // Create an ImageServer where the pixels are derived only from annotations
        LabeledImageServer labelServer = new LabeledImageServer.Builder(imageData)
                .backgroundLabel(0, ColorTools.WHITE)
                .downsample(1.0)
                .addLabel("Gel Band", 1)
                .multichannelOutput(false)
                .build();

        // saves the image
        writeImage(labelServer, filename);
    }
}
