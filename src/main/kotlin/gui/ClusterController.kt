package gui

import clustering.Clusterer
import clustering.LeafNode
import clustering.drawRect
import javafx.embed.swing.SwingFXUtils
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.rendering.PDFRenderer
import tornadofx.*
import java.awt.image.BufferedImage
import java.io.File


class ClusterController: Controller() {
    private val model: StateModel by inject()
    private val clusterer = Clusterer()

    fun cluster() {
        model.running.value = true
        var image: BufferedImage? = null

        runAsync {
            model.apply {
                val doc = PDDocument.load(File(item.path))
                clusterer.vectorizer = item.vectorizer
                dendrogram.value = clusterer.clusterFilePage(doc, item.pagenum)

                image = PDFRenderer(doc).renderImage(item.pagenum)
                doc.close()
            }
        } ui {
            model.image.value = SwingFXUtils.toFXImage(image, null)
            model.running.value = false
            model.commit()
        }
    }

    fun cluster_dbscan() {
        model.running.value = true
        var image: BufferedImage? = null

        runAsync {
            model.apply {
                val doc = PDDocument.load(File(item.path))
                val page = doc.getPage(item.pagenum)
                clusterer.vectorizer = item.vectorizer
                val merged = clusterer.clusterFilePageDbscan(doc, item.pagenum, item.epsilon, item.minSamples)
                val bboxes = merged.map(clusterer::getBoundingRect)
                bboxes.forEach { doc.drawRect(page, it) }

                image = PDFRenderer(doc).renderImage(item.pagenum)
                doc.close()

                blocks.value = merged.observable()
            }
        } ui {
            model.image.value = SwingFXUtils.toFXImage(image, null)
            model.running.value = false
            model.commit()
        }
    }

    fun merge() {
        model.running.value = true
        var image: BufferedImage? = null

        runAsync {
            model.apply {
                val doc = PDDocument.load(File(item.path))
                val page = doc.getPage(item.pagenum)
                val merged = item.collector.function(item.dendrogram!!, item.threshold)
                val bboxes = merged.map(clusterer::getBoundingRect)
                bboxes.forEach { doc.drawRect(page, it) }

                image = PDFRenderer(doc).renderImage(item.pagenum)
                doc.close()

                blocks.value = merged.observable()
            }
        } ui {
            model.image.value = SwingFXUtils.toFXImage(image, null)
            model.running.value = false
            model.merged.value = true
            model.commit()
        }
    }

    fun kmeans() {
        val centroids = clusterer.clusterDistances(model.item.dendrogram!!, model.item.threshold)
        println(centroids)
    }

    fun recluster() {
        model.running.value = true
        var image: BufferedImage? = null

        runAsync {
            model.apply {
                val doc = PDDocument.load(File(item.path))
                clusterer.vectorizer = item.vectorizer
                dendrogram.value = clusterer.recluster(item.blocks!!)

                image = PDFRenderer(doc).renderImage(item.pagenum)
                doc.close()
            }
        } ui {
            model.image.value = SwingFXUtils.toFXImage(image, null)
            model.running.value = false
            model.commit()
        }
    }

    fun labelClusters() {
        clusterer.labelClusters(model.item.blocks!!, 2)
    }
}
