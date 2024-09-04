/*
 * Copyright 2023/24 University of Edinburgh
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * Copied with attribution from qupath-extension-wsinfer.
 * Further edits were made to this file for the GelGenie qupath extension, also licensed under Apache 2.0.
 */

package qupath.ext.gelgenie.models;

import javafx.util.StringConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.UserDirectoryManager;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.io.GsonTools;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;

public class ModelInterfacing {
    private static final Logger logger = LoggerFactory.getLogger(ModelInterfacing.class);
    private static GelGenieModelCollection cachedModelCollection;

    /**
     * Loads a named gelgenie model from the available collection.
     * @param modelName: String name of the model to be loaded
     * @return The model, in the wrapper class format.
     * @throws IllegalArgumentException
     */
    public static GelGenieModel loadModel(String modelName) throws IllegalArgumentException {
        Objects.requireNonNull(modelName, "Model name cannot be null");
        var modelCollection = getModelCollection();
        var model = modelCollection.getModels().getOrDefault(modelName, null);
        if (model == null) {
            throw new IllegalArgumentException("No model found with name: " + modelName);
        }
        return model;
    }

    /**
     * Gets the list of models available for downloading.  This might require getting the list from HuggingFace.
     * @return
     */
    public static GelGenieModelCollection getModelCollection() {
        if (cachedModelCollection == null) {
            synchronized (ModelInterfacing.class) {
                if (cachedModelCollection == null)
                    cachedModelCollection = downloadModelCollection();
            }
        }
        return cachedModelCollection;
    }

    /**
     * Downloads the model collection from the HuggingFace repo.
     * @return The collection of models, in the wrapper class format.
     */
    private static GelGenieModelCollection downloadModelCollection() {
        String json;
        URL url = null;
        try {
            url = new URL("https://huggingface.co/datasets/mattaq/GelGenie-Model-Zoo/raw/main/registry.json");
        } catch (MalformedURLException e) {
            logger.error("Malformed URL", e);
        }
        Path cachedFile = Paths.get(getUserDirectory(), "registry.json");

        try {
            checkPathExists(Path.of(getUserDirectory()));
            downloadURLToFile(url, cachedFile.toFile());
            logger.info("Downloaded zoo file {}", cachedFile);
        } catch (IOException e) {
            logger.error("Unable to download zoo JSON file {}", cachedFile, e);
        }
        try {
            json = Files.readString(cachedFile);
            logger.info("Read cached zoo file {}", cachedFile);
        } catch (IOException e) {
            logger.error("Unable to read cached zoo JSON file", e);
            return null;
        }
        return GsonTools.getInstance().fromJson(json, GelGenieModelCollection.class);
    }

    /**
     * Gets the default directory where all common records and models can be saved for any user.
     * @return the path to the gelgenie directory
     */
    public static String getUserDirectory(){
        String userPath = String.valueOf(UserDirectoryManager.getInstance().getUserPath());
        return Paths.get(userPath, "gelgenie").toString();
    }

    /**
     * Downloads a file from a specific URL.
     * @param url The url containing the required data
     * @param file The file location to which the data will be downloaded to
     * @throws IOException
     */
    static void downloadURLToFile(URL url, File file) throws IOException {
        try (InputStream stream = url.openStream()) {
            try (ReadableByteChannel readableByteChannel = Channels.newChannel(stream)) {
                try (FileOutputStream fos = new FileOutputStream(file)) {
                    fos.getChannel().transferFrom(readableByteChannel, 0, Long.MAX_VALUE);
                }
            }
        }
    }
    /**
     * Check if a directory exists and create it if it does not.
     * @param path the path of the directory
     * @return true if the directory exists when the method returns
     */
    public static boolean checkPathExists(Path path) {
        if (!path.toFile().exists()) {
            try {
                Files.createDirectories(path);
            } catch (IOException e) {
                logger.error("Cannot create directory {}", path, e);
                return false;
            }
        }
        return true;
    }

    /**
     * Wrapper for reading in the main JSON file containing the list of available gelgenie models.
     */
    public class GelGenieModelCollection {
        private Map<String, GelGenieModel> models;

        /**
         * Get a map of model names to models.
         */
        public Map<String, GelGenieModel> getModels() {
            return Collections.unmodifiableMap(models);
        }

        public void addModel(String name, GelGenieModel model) {
            models.put(name, model);
        }
    }

    /**
     * Wrapper which converts the selected model into a string descriptor (for use in the dropdown menu).
     */
    public static class ModelStringConverter extends StringConverter<GelGenieModel> {
        private final GelGenieModelCollection models;

        public ModelStringConverter(GelGenieModelCollection models) {
            Objects.requireNonNull(models, "Models cannot be null");
            this.models = models;
        }

        @Override
        public String toString(GelGenieModel object) {
            for (var entry : models.getModels().entrySet()) {
                if (entry.getValue() == object)
                    return entry.getValue().getAbbrvName();
            }
            return "";
        }

        @Override
        public GelGenieModel fromString(String string) {
            return models.getModels().getOrDefault(string, null);
        }
    }
}
