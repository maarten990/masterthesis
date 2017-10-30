package gui

import clustering.Clusterer
import clustering.drawRect
import javafx.embed.swing.SwingFXUtils
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.rendering.PDFRenderer
import tornadofx.*
import java.awt.image.BufferedImage
import java.io.File


class ClusterController: Controller() {
    private val param: ParamsModel by inject()
    private val mergeParam: MergeParamsModel by inject()
    private val status: StatusModel by inject()
    private val results: ResultsModel by inject()
    private val clusterer = Clusterer()

    fun cluster() {
        status.running.value = true
        var image: BufferedImage? = null

        runAsync {
            val doc = PDDocument.load(File(param.item.path))
            clusterer.vectorizer = param.item.vectorizer
            results.clusters.value = clusterer.clusterFilePage(doc, param.item.pagenum)

            image = PDFRenderer(doc).renderImage(param.item.pagenum)
            doc.close()
        } ui {
            results.image.value = SwingFXUtils.toFXImage(image, null)
            results.commit()
            status.running.value = false
        }
    }

    fun merge() {
        status.running.value = true
        var image: BufferedImage? = null

        runAsync {
            val doc = PDDocument.load(File(param.item.path))
            val page = doc.getPage(param.item.pagenum)
            val merged = mergeParam.item.collector.function(results.item.clusters, mergeParam.item.threshold)
            val bboxes = merged.map(clusterer::getBoundingRect)
            bboxes.forEach { doc.drawRect(page, it) }

            image = PDFRenderer(doc).renderImage(param.item.pagenum)
            doc.close()

            results.merged.value = merged.observable()
        } ui {
            results.image.value = SwingFXUtils.toFXImage(image, null)
            results.commit()

            status.running.value = false
            status.merged.value = true
        }
    }

    fun kmeans() {
        val centroids = clusterer.clusterDistances(results.item.clusters, mergeParam.item.threshold)
        println(centroids)
    }

    fun recluster() {
        status.running.value = true
        var image: BufferedImage? = null

        runAsync {
            val doc = PDDocument.load(File(param.item.path))
            clusterer.vectorizer = param.item.vectorizer
            results.clusters.value = clusterer.recluster(results.item.merged)

            image = PDFRenderer(doc).renderImage(param.item.pagenum)
            doc.close()
        } ui {
            results.image.value = SwingFXUtils.toFXImage(image, null)
            results.commit()
            status.running.value = false
        }
    }

    fun labelClusters() {
        val clusters = mergeParam.item.collector.function(results.item.clusters, mergeParam.item.threshold)
        clusterer.labelClusters(clusters, 2)
    }
}
