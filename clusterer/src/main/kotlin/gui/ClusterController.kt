package gui

import clustering.*
import javafx.embed.swing.SwingFXUtils
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.rendering.PDFRenderer
import tornadofx.*
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File

val COLORS = listOf(Color.RED, Color.BLUE, Color.GREEN, Color.MAGENTA, Color.ORANGE, Color.CYAN, Color.YELLOW)

class ClusterController: Controller() {
    private val model: StateModel by inject()
    private val clusterer = Clusterer()

    fun cluster() {
        model.progress.value = 0.0f

        runAsync {
            val item = model.item
            val doc = PDDocument.load(File(item.path))
            val dendrograms = mutableListOf<Dendrogram>()

            for (pagenum in 0 until doc.numberOfPages) {
                model.progress.value = pagenum.toFloat() / doc.numberOfPages.toFloat()
                clusterer.vectorizer = item.vectorizer
                dendrograms.add(clusterer.clusterFilePage(doc, item.pagenum))
            }

            model.dendrogram.value = dendrograms.observable()
            doc.close()
        } ui {
            model.progress.value = 1.0f
            model.commit()
        }
    }

    fun cluster_dbscan() {
        model.progress.value = 0.0f

        runAsync {
            val item = model.item
            val merged = mutableListOf<Map<CharData, Int>>()
            val doc = PDDocument.load(File(item.path))

            for (pagenum in 0 until doc.numberOfPages) {
                model.progress.value = pagenum.toFloat() / doc.numberOfPages.toFloat()
                clusterer.vectorizer = item.vectorizer
                merged.add(clusterer.clusterFilePageDbscan(doc, pagenum, item.epsilon, item.minSamples))
            }

            doc.close()
            model.blocks.value = merged.observable()
        } ui {
            model.progress.value = 1.0f
            model.commit()
            drawBlocks()
        }
    }

    fun merge() {
        model.progress.value = 0.0f

        runAsync {
            val item = model.item
            model.blocks.value = item.dendrogram.map { item.collector.function(it, item.threshold) }.observable()
        } ui {
            model.progress.value = 1.0f
            model.merged.value = true
            model.commit()
            drawBlocks()
        }
    }

    /*
    fun kmeans() {
        val centroids = clusterer.clusterDistances(model.item.dendrogram, model.item.threshold)
        println(centroids)
    }
    */

    fun recluster() {
        /*
        model.progress.value = 0.0f

        runAsync {
            val item = model.item
            clusterer.vectorizer = item.vectorizer
            model.dendrogram.value = item.blocks.map { clusterer.recluster(it) }.observable()
        } ui {
            model.progress.value = 1.0f
            model.commit()
        }
        */
    }

    fun labelClusters() {
        model.progress.value = 0.0f
        val colormap = mutableMapOf<CharData, Color>()

        runAsync {
            val item = model.item
            clusterer.vectorizer = item.kVect

            // group the data into lists of chardata objects belonging to the same cluster, for the document as a whole
            val clusterGroups = item.blocks.flatMap { labelMappingToLists(it) }

            val labeled = when (model.item.labeler) {
                BlockLabeler.KMEANS -> clusterer.kmeans(clusterGroups.map(::getBoundingRect), model.item.k)
                BlockLabeler.DBSCAN -> clusterer.dbscan(clusterGroups.map(::getBoundingRect), model.item.epsilon, model.item.minSamples)
                null -> mapOf()
            }

            // create a colormap
            for ((color, clusters) in COLORS.zip(labelMappingToLists(labeled))) {
                clusters.forEach { colormap[it] = color }
            }
        } ui {
            model.progress.value = 1.0f
            model.colormap.value = colormap.observable()
            model.commit()
            drawBlocks()
        }
    }

    fun drawBlocks() {
        if (model.item.blocks == null) {
            return
        }

        var image: BufferedImage? = null
        val doc = PDDocument.load(File(model.item.path))

        runAsync {
            val page = doc.getPage(model.item.pagenum)
            val bboxes = labelMappingToLists(model.item.blocks[model.item.pagenum]).map(::getBoundingRect)
            bboxes.forEach { doc.drawRect(page, it, color = model.item.colormap?.get(it) ?: Color.BLACK) }
            image = PDFRenderer(doc).renderImage(model.item.pagenum)
            doc.close()
        } ui {
            if (image != null)
                model.image.value = SwingFXUtils.toFXImage(image, null)
            else
                println("Warning: could not render image")

            model.commit()
        }
    }
}

/**
 * Convert a mapping to cluster labels to a list of items belong to the same clustering. For example:
 * Input: {a: 1, b: 1, c: 2}
 * Output: [[a, b], [c]]
 */
fun labelMappingToLists(mapping: Map<CharData, Int>): List<List<CharData>> {
    val clusterGroups = mutableMapOf<Int, MutableList<CharData>>()
    for ((data, id) in mapping) {
        if (clusterGroups.containsKey(id)) {
            clusterGroups[id]?.add(data)
        } else {
            clusterGroups[id] = mutableListOf(data)
        }
    }

    return clusterGroups.values.toList()
}
