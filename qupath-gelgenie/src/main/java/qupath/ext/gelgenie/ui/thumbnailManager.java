// all of this code has been replicated from QuPath's 'PathObjectImageManagers' as that class is not public

package qupath.ext.gelgenie.ui;

import javafx.beans.property.DoubleProperty;
import javafx.beans.value.WeakChangeListener;
import javafx.concurrent.Task;
import javafx.concurrent.WorkerStateEvent;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.EventHandler;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.canvas.Canvas;
import javafx.scene.control.ContentDisplay;
import javafx.scene.control.TableCell;
import javafx.scene.image.Image;
import javafx.scene.text.TextAlignment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.images.stores.DefaultImageRegionStore;
import qupath.lib.gui.images.stores.ImageRenderer;
import qupath.lib.gui.tools.GuiTools;
import qupath.lib.gui.viewer.OverlayOptions;
import qupath.lib.gui.viewer.PathObjectPainter;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.regions.RegionRequest;

import java.awt.image.BufferedImage;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;

public class thumbnailManager {
    public static class thumbnailTableCell<S extends BandEntry, T extends PathObject> extends TableCell<S, T> {
        private static final Logger logger = LoggerFactory.getLogger(thumbnailTableCell.class);
        private PathObjectViewer<Canvas> painter;
        private Canvas canvas = new Canvas();
        private double preferredSize = 100;

    //		private static AtomicInteger counter = new AtomicInteger();

        thumbnailTableCell(PathObjectViewer<Canvas> painter, double padding) {
            super();
    //			System.err.println("New cell: " + counter.incrementAndGet());
            logger.trace("Creating new cell ({})", + System.identityHashCode(this));
            this.setContentDisplay(ContentDisplay.CENTER);
            this.setAlignment(Pos.CENTER);
            this.setTextAlignment(TextAlignment.CENTER);
            this.painter = painter;
            this.canvas = painter.getNode();
            canvas.setWidth(preferredSize);
            canvas.setHeight(preferredSize);
            canvas.setStyle("-fx-effect: dropshadow(three-pass-box, rgba(0,0,0,0.5), 4, 0, 1, 1);");
            canvas.heightProperty().bind(canvas.widthProperty());

            tableColumnProperty().addListener((v, o, n) -> {
                canvas.widthProperty().unbind();
                if (n != null)
                    canvas.widthProperty().bind(n.widthProperty().subtract(padding*2));
            });

            // Update image on width changes
            // Required to handle the fact that object ROIs may need to be repainted
            // with updated stroke widths
            canvas.widthProperty().addListener((v, o, n) -> painter.updateImage(false));
        }


        @Override
        protected void updateItem(T pathObject, boolean empty) {
            super.updateItem(pathObject, empty);
            setText(null);
            if (empty || pathObject == null || pathObject.getROI() == null) {
                setGraphic(null);
                return;
            }
            setGraphic(canvas);
            painter.setPathObject(pathObject);
        }

    }

    public static <S extends BandEntry, T extends PathObject> TableCell<S, T> createTableCell(QuPathViewer viewer, ImageServer<BufferedImage> server, boolean paintObject, double padding, ExecutorService pool) {
        var painter = createCanvasPainter(viewer, server, paintObject, pool);
        return new thumbnailTableCell<>(painter, padding);
    }

    public static PathObjectViewer<Canvas> createCanvasPainter(QuPathViewer viewer, ImageServer<BufferedImage> server, boolean paintObject, ExecutorService pool) {
        return new PathObjectCanvasManager(new Canvas(), viewer, server, paintObject, pool);
    }

    static class PathObjectCanvasManager extends PathObjectViewer<Canvas> {

        PathObjectCanvasManager(Canvas canvas, QuPathViewer viewer, ImageServer<BufferedImage> server, boolean paintObject, ExecutorService pool) {
            super(canvas, viewer, server, paintObject,
                    canvas.widthProperty(), canvas.heightProperty(), pool);
        }

        @Override
        protected void updateNode(Image image) {
            var canvas = getNode();
            if (image == null)
                clearCanvas(canvas);
            else {
                GuiTools.paintImage(canvas, image);
            }
        }

    }

    private static void clearCanvas(Canvas canvas) {

        canvas.getGraphicsContext2D().clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
    }
    public abstract static class PathObjectViewer<T extends Node> {

        private static final Logger logger = LoggerFactory.getLogger(PathObjectViewer.class);

        private QuPathViewer viewer;
        private boolean paintObject;

        private ImageServer<BufferedImage> server;

        private PathObject pathObject;

        private T node;
        private DoubleProperty widthProperty;
        private DoubleProperty heightProperty;

        private PathObjectThumbnailTask task;
        private PathObjectViewer.WorkerHandler handler = new PathObjectViewer.WorkerHandler();

        private int maxCacheSize = 100;

        private ExecutorService pool;

        private WeakChangeListener<Number> numberListenerFalse = new WeakChangeListener<>((v, o, n) -> updateImage(false));
        private WeakChangeListener<Number> numberListenerTrue = new WeakChangeListener<>((v, o, n) -> updateImage(true));

        private Map<PathObject, Image> cache = Collections.synchronizedMap(new LinkedHashMap<>() {

            private static final long serialVersionUID = 1L;

            @Override
            protected boolean removeEldestEntry(Map.Entry<PathObject, Image> eldest) {
                return size() > maxCacheSize;
            }

        });

        PathObjectViewer(T node, QuPathViewer viewer, ImageServer<BufferedImage> server, boolean paintObject,
                         DoubleProperty widthProperty, DoubleProperty heightProperty, ExecutorService pool) {
            logger.trace("Creating new cell ({})", + System.identityHashCode(this));
            this.server = server;
            this.viewer = viewer;
            this.paintObject = paintObject;
            this.pool = pool == null ? ForkJoinPool.commonPool() : pool;
            if (paintObject && viewer != null) {
                var options = viewer.getOverlayOptions();
                options.lastChangeTimestamp().addListener(numberListenerFalse);
                viewer.getImageDisplay().changeTimestampProperty().addListener(numberListenerFalse);
            }
            this.node = node;
            this.widthProperty = widthProperty;
            this.heightProperty = heightProperty;
            widthProperty.addListener(numberListenerTrue);
            heightProperty.addListener(numberListenerTrue);
            node.visibleProperty().addListener(new WeakChangeListener<>((v, o, n) -> updateImage(true)));
        }

        public void setPathObject(PathObject pathObject) {
            if (this.pathObject == pathObject)
                return;
            if (task != null) {
                this.task.cancel(true);
                this.task = null;
            }
            this.pathObject = pathObject;
            updateImage(false);
        }

        public PathObject getPathObject() {
            return pathObject;
        }

        public T getNode() {
            return node;
        }

        public void resetCache() {
            cache.clear();
        }

        public int getMaxCacheSize() {
            return maxCacheSize;
        }

        public void setMaxCacheSize(int maxSize) {
            maxCacheSize = maxSize;
            if (cache.size() > maxCacheSize) {
                synchronized (cache) {
                    var iterator = cache.entrySet().iterator();
                    while (iterator.hasNext() && cache.size() > maxCacheSize) {
                        iterator.next();
                        iterator.remove();
                    }
                }
            }
        }


        @SuppressWarnings("unchecked")
        public void updateImage(boolean useCache) {
            var roi = pathObject == null ? null : pathObject.getROI();
            if (roi == null || !node.isVisible()) {
                return;
            }
            try {
                int width = (int)Math.round(widthProperty.get());
                int height = (int)Math.round(heightProperty.get());

                Image image = null;
                if (useCache && task != null && task.isDone()) {
                    image = cache.get(pathObject);
                    if (image == null && task != null && task.isDone())
                        image = task.getValue();
                    if (image != null && image.getWidth() == width && image.getHeight() == height) {
                        updateNode(image);
                        return;
                    } else if (image != null) {
//						updateNode(image);
                    }
                }
                // Don't draw on a tiny canvas
                if (isTooSmall(width, height)) {
                    updateNode(null);
                    return;
                }
                if (task != null)
                    task.cancel(true);

                if (viewer != null)
                    task = new PathObjectThumbnailTask(server, pathObject, paintObject, width, height, viewer);
                else
                    task = new PathObjectThumbnailTask(server, pathObject, paintObject, width, height);

                task.setOnSucceeded(handler);
                task.setOnFailed(handler);

                pool.submit(task);
            } catch (Exception e) {
                logger.error("Problem reading thumbnail for {}: {}", pathObject, e);
            }
        }

        protected boolean isTooSmall(int width, int height) {
            return width <= 2 || height <= 2;
        }

        protected abstract void updateNode(Image image);


        class WorkerHandler implements EventHandler<WorkerStateEvent> {

            @Override
            public void handle(WorkerStateEvent event) {
                if (event.getEventType() == WorkerStateEvent.WORKER_STATE_SUCCEEDED) {
                    try {
                        var image = task.getValue();
                        if (image != null) {
                            cache.put(task.getPathObject(), image);
                            updateNode(image);
                        }
                        event.consume();
                        return;
                    } catch (Exception e) {
                        logger.debug(e.getLocalizedMessage(), e);
                    }
                }
                updateNode(null);
                event.consume();
            }

        }

    }

    static class PathObjectThumbnailTask extends Task<Image> {

        private static final Logger logger = LoggerFactory.getLogger(PathObjectThumbnailTask.class);

        private DefaultImageRegionStore store;
        private ImageRenderer renderer;
        private OverlayOptions options;

        private ImageServer<BufferedImage> server;
        private PathObject pathObject;
        private int width, height;
        private boolean paintObject;

        PathObjectThumbnailTask(ImageServer<BufferedImage> server, PathObject pathObject, boolean paintObject, int width, int height) {
            this(server, pathObject, paintObject, width, height, null, null, null);
        }

        PathObjectThumbnailTask(ImageServer<BufferedImage> server, PathObject pathObject, boolean paintObject, int width, int height, QuPathViewer viewer) {
            this(server, pathObject, paintObject, width, height, viewer.getImageRegionStore(), viewer.getImageDisplay(), viewer.getOverlayOptions());
        }

        PathObjectThumbnailTask(ImageServer<BufferedImage> server, PathObject pathObject, boolean paintObject, int width, int height,
                                DefaultImageRegionStore store, ImageRenderer renderer, OverlayOptions options) {
            super();
            this.store = store;
            this.renderer = renderer;
            this.options = options;

            this.server = server;
            this.pathObject = pathObject;
            this.paintObject = paintObject;
            this.width = width;
            this.height = height;
        }

        public PathObject getPathObject() {
            return pathObject;
        }

        @Override
        public Image call() {

            if (Thread.interrupted() || pathObject == null || !pathObject.hasROI())
                return null;

            var roi = pathObject.getROI();
            double scaleFactor = 1.2; // Used to give a bit more context around the ROI
            double downsample = Math.max(roi.getBoundsWidth() * scaleFactor / width, roi.getBoundsHeight() * scaleFactor / height);
            if (downsample < 1)
                downsample = 1;

            try {
                BufferedImage img;
                if (store != null) {
                    img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                    var g2d = img.createGraphics();
                    g2d.setClip(0, 0, img.getWidth(), img.getHeight());
                    g2d.scale(1.0/downsample, 1.0/downsample);

                    double x = roi.getCentroidX() - img.getWidth() / 2.0 * downsample;
                    double y = roi.getCentroidY() - img.getHeight() / 2.0 * downsample;
                    g2d.translate(-x, -y);

                    store.paintRegionCompletely(server, g2d, g2d.getClipBounds(), roi.getZ(), roi.getT(), downsample, null, renderer, 500L);
                    if (paintObject && options != null) {
                        PathObjectPainter.paintObject(pathObject, g2d, options, null, downsample);
                    }
                    g2d.dispose();
                } else {
                    RegionRequest request = RegionRequest.createInstance(server.getPath(), downsample, roi);
                    img = server.readRegion(request);
                }

                return SwingFXUtils.toFXImage(img, null);
            } catch (Exception e) {
                logger.error("Unable to draw image for " + pathObject, e);
            }
            return null;
        }
    }
}