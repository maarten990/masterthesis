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
            param.run {
                val doc = PDDocument.load(File(path.value))
                results.clusterer.vectorizer = vectorizer.value
                results.clusters.value = results.clusterer.clusterFilePage(doc, pagenum.value.toInt())

                image = PDFRenderer(doc).renderImage(pagenum.value.toInt())
                doc.close()
            }
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
            mergeParam.run {
                val doc = PDDocument.load(File(param.path.value))
                val page = doc.getPage(param.pagenum.value.toInt())
                val merged = collector.value.function(results.clusters.value, threshold.value.toInt())
                val bboxes = merged.map(results.clusterer::getBoundingRect)
                bboxes.forEach { drawRect(doc, page, it) }

                image = PDFRenderer(doc).renderImage(param.pagenum.value.toInt())
                doc.close()
            }
        } ui {
            results.image.value = SwingFXUtils.toFXImage(image, null)
            results.commit()
            status.running.value = false
        }
    }
}
