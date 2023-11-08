// This was mostly edited from the WSInfer extension - need to check with Pete how to properly attribute

package qupath.ext.gelgenie.models;

import com.google.gson.annotations.SerializedName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.io.GsonTools;

import java.io.File;
import java.io.IOException;
import java.math.BigInteger;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import static qupath.ext.gelgenie.models.ModelInterfacing.checkPathExists;

public class GelGenieModel {

    private static final Logger logger = LoggerFactory.getLogger(GelGenieModel.class);

    private String description;

    @SerializedName("hf_repo_id")
    private String hfRepoId;

    @SerializedName("hf_revision")
    private String hfRevision;

    @SerializedName("onnx_file")
    private String onnxModel;

    @SerializedName("torchscript_file")
    private String torchscriptModel;

    @SerializedName("abbrvName")
    private String abbrvName;

    public String getName() {
        return hfRepoId;
    }

    public String getAbbrvName() {return abbrvName;}

    /**
     * Remove the cached model files.
     */
    public synchronized void removeCache() {
        getTSFile().delete();
        getCFFile().delete();
        getOnnxFile().delete();
    }

    /**
     * Get the torchscript file. Note that it is not guaranteed that the model has been downloaded.
     * @return path to torchscript pt file in cache dir
     */
    public File getTSFile() {
        return getFile(String.format("torchscript_checkpoints/%s", torchscriptModel));
    }

    public File getOnnxFile() {
        return getFile(String.format("onnx_checkpoints/%s", onnxModel));
    }

    /**
     * Get the configuration file. Note that it is not guaranteed that the model has been downloaded.
     * @return path to model config file in cache dir
     */
    public File getCFFile() {
        return getFile("config.toml");
    }

    /**
     * Check if the model files exist and are valid.
     * @return true if the files exist and the SHA matches, and the config is valid.
     */
    public boolean isValid() {
        return getTSFile().exists() && getOnnxFile().exists() && checkModifiedTimes();
    }

    /**
     * Check if the LFS pointer that contains the SHA has later modified time
     * than the model file. This should always be true since we download the
     * model first.  This is only applied only the TS file currently.  TODO: extend to ONNX too
     * @return true if the modified times are as expected.
     */
    private boolean checkModifiedTimes() {
        try {
            return Files.getLastModifiedTime(getTSFile().toPath())
                    .compareTo(Files.getLastModifiedTime(getPointerFileTS().toPath())) < 0;
        } catch (IOException e) {
            logger.error("Cannot get last modified time");
            return false;
        }
    }

    private File getPointerFileTS() {
        return getFile("lfs-pointer-ts.txt");
    }
    private File getPointerFileOnnx() {
        return getFile("lfs-pointer-onnx.txt");
    }

    private File getFile(String f) {
        return Paths.get(getModelDirectory().toString(), f).toFile();
    }

    private File getModelDirectory() {
        return Paths.get(ModelInterfacing.getUserDirectory(), hfRepoId, hfRevision).toFile();
    }


    private static String checkSumSHA256(File file) throws IOException, NoSuchAlgorithmException {
        byte[] data = Files.readAllBytes(file.toPath());
        byte[] hash = MessageDigest.getInstance("SHA-256").digest(data);
        return new BigInteger(1, hash).toString(16);
    }

    /**
     * Check that the SHA-256 checksum in the LFS pointer file matches one
     * we calculate ourselves.
     * @return true if the torchscript and onnx model files are identical to the remote ones.
     */
    private boolean checkSHAMatches() {
        try { //TODO: make this more elegant.
            String shaDown = checkSumSHA256(getTSFile());
            String content = Files.readString(getPointerFileTS().toPath(), StandardCharsets.UTF_8);
            String shaUp = content.split("\n")[1].replace("oid sha256:", "");
            if (!shaDown.equals(shaUp)) {
                return false;
            }
            String shaDownonnx = checkSumSHA256(getOnnxFile());
            String contentonnx = Files.readString(getPointerFileOnnx().toPath(), StandardCharsets.UTF_8);
            String shaUponnx = contentonnx.split("\n")[1].replace("oid sha256:", "");
            if (!shaDownonnx.equals(shaUponnx)) {
                return false;
            }
        } catch (IOException | NoSuchAlgorithmException e) {
            logger.error("Unable to generate SHA for {}", getTSFile(), e);
            return false;
        }
        return true;
    }

    /**
     * Request that the model is downloaded.
     */
    public synchronized void downloadModel() throws IOException {
        File modelDirectory = getModelDirectory();
        if (!modelDirectory.exists()) {
            Files.createDirectories(modelDirectory.toPath());
        }
        checkPathExists(Paths.get(modelDirectory.toString(), "onnx_checkpoints"));
        checkPathExists(Paths.get(modelDirectory.toString(), "torchscript_checkpoints"));

        downloadFileToCacheDir(String.format("onnx_checkpoints/%s", onnxModel));
        downloadFileToCacheDir(String.format("torchscript_checkpoints/%s", torchscriptModel));
        downloadFileToCacheDir("config.toml");

        URL urlPt1 = new URL(String.format("https://huggingface.co/%s/raw/%s/onnx_checkpoints/%s", hfRepoId, hfRevision, onnxModel));
        ModelInterfacing.downloadURLToFile(urlPt1, getPointerFileOnnx());

        URL urlPt2 = new URL(String.format("https://huggingface.co/%s/raw/%s/torchscript_checkpoints/%s", hfRepoId, hfRevision, torchscriptModel));
        ModelInterfacing.downloadURLToFile(urlPt2, getPointerFileTS());

        if (!isValid() || !checkSHAMatches()) {
            throw new IOException("Error downloading model files");
        }
    }

    private void downloadFileToCacheDir(String file) throws IOException {
        URL url = new URL(String.format("https://huggingface.co/%s/resolve/%s/%s", hfRepoId, hfRevision, file));
        ModelInterfacing.downloadURLToFile(url, getFile(file));
    }


}
