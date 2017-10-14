package gui

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

    fun cluster() {
        status.running.value = true
        var image: BufferedImage? = null

        runAsync {
            val doc = PDDocument.load(File(param.item.path))
            results.clusterer.vectorizer = param.item.vectorizer
            results.clusters.value = results.clusterer.clusterFilePage(doc, param.item.pagenum)

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
            val bboxes = merged.map(results.clusterer::getBoundingRect)
            bboxes.forEach { drawRect(doc, page, it) }

            image = PDFRenderer(doc).renderImage(param.item.pagenum)
            doc.close()
        } ui {
            results.image.value = SwingFXUtils.toFXImage(image, null)
            results.commit()
            status.running.value = false
        }
    }
}
