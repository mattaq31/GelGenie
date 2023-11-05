// This code was extracted from the WSInfer extension.
package qupath.ext.gelgenie.models;

import ai.djl.engine.Engine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.common.GeneralTools;

import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.concurrent.Callable;

/**
 * Helper class to manage access to PyTorch via Deep Java Library.
 */
class PytorchManager {

    private static final Logger logger = LoggerFactory.getLogger(PytorchManager.class);

    /**
     * Get the available devices for PyTorch, including MPS if Apple Silicon.
     * @return
     */
    static Collection<String> getAvailableDevices() {
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
        // CPU should always be available
        if (availableDevices.isEmpty())
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
    static boolean hasPyTorchEngine() {
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
