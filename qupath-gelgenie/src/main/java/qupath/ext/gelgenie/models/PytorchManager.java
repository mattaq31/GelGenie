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
// This code was extracted from the WSInfer extension.
package qupath.ext.gelgenie.models;

import ai.djl.Device;
import ai.djl.engine.Engine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.gelgenie.ui.GelGeniePrefs;
import qupath.lib.common.GeneralTools;

import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.concurrent.Callable;

/**
 * Helper class to manage access to PyTorch via Deep Java Library.
 */
public class PytorchManager {

    private static final Logger logger = LoggerFactory.getLogger(PytorchManager.class);

    public static Device getDevice() {
        String deviceName = GelGeniePrefs.deviceProperty().get();
        switch (deviceName) {
            case "gpu":
                logger.info("Using GPU");
                return Device.gpu();
            case "cpu":
                logger.info("Using CPU");
                return Device.cpu();
            default:
                logger.info("Attempting to set device to {}", deviceName);
                return Device.fromName(deviceName);
        }
    }

    /**
     * Check if a specified device corresponds to using the Metal Performance Shaders (MPS) backend (Apple Silicon)
     * @param device
     * @return
     */
    public static boolean isMPS(Device device) {
        return device != null && device.getDeviceType().toLowerCase().startsWith("mps");
    }

    /**
     * Get the available devices for PyTorch, including MPS if Apple Silicon.
     * @return
     */
    public static Collection<String> getAvailableDevices() {
        Set<String> availableDevices = new LinkedHashSet<>();
        boolean includesMPS = false; // Don't add MPS twice
        var engine = getEngineOffline();
        if (engine != null) {
            for (var device : engine.getDevices()) {
                String name = device.getDeviceType();
                availableDevices.add(name);
                if (name.toLowerCase().startsWith("mps"))
                    includesMPS = true;
            }
        }
        // CPU should always be available, so we don't need to check
//        if (availableDevices.isEmpty())
        availableDevices.add("cpu");

        // If we could use MPS, but don't have it already, add it
        if (!includesMPS && GeneralTools.isMac() && "aarch64".equals(System.getProperty("os.arch"))) {
            availableDevices.add("mps");
        }
        return availableDevices;
    }

    /**
     * Query if the PyTorch engine is already available, without a need to downlaod.
     * @return
     */
    public static boolean hasPyTorchEngine() {
        return getEngineOffline() != null;
    }

    /**
     * Get the PyTorch engine, without automatically downloading it if it isn't available.
     * @return the engine if available, or null otherwise
     */
    static Engine getEngineOffline() {
        try {
            return callOffline(() -> Engine.getEngine("PyTorch"));
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
            return null;
        }
    }

    /**
     * Get the PyTorch engine, downloading if necessary.
     * @return the engine if available, or null if this failed
     */
    static Engine getEngineOnline() {
        try {
            return callOnline(() -> Engine.getEngine("PyTorch"));
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
            return null;
        }
    }

    /**
     * Call a function with the "offline" property set to true (to block automatic downloads).
     * @param callable
     * @return
     * @param <T>
     * @throws Exception
     */
    private static <T> T callOffline(Callable<T> callable) throws Exception {
        return callWithTempProperty("offline", "true", callable);
    }

    /**
     * Call a function with the "offline" property set to false (to allow automatic downloads).
     * @param callable
     * @return
     * @param <T>
     * @throws Exception
     */
    private static <T> T callOnline(Callable<T> callable) throws Exception {
        return callWithTempProperty("offline", "false", callable);
    }


    private static <T> T callWithTempProperty(String property, String value, Callable<T> callable) throws Exception {
        String oldValue = System.getProperty(property);
        System.setProperty(property, value);
        try {
            return callable.call();
        } finally {
            if (oldValue == null)
                System.clearProperty(property);
            else
                System.setProperty(property, oldValue);
        }
    }


}
