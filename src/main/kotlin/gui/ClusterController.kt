package gui

/*
import clustering.Clusterer
import clustering.collectBelowCutoff
import clustering.drawRect
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.rendering.PDFRenderer
import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import javax.swing.ImageIcon
import kotlin.concurrent.thread

class ClusterController(val view: ClusterView) {
    val model = ClusterModel(10, Vectorizer.ALL, 0, "", null)
    val clusterer = Clusterer()

    var threshold: String
        get() = model.threshold.toString()
        set(value) {
            val int = value.toIntOrNull()
            model.threshold = int ?: -1
        }

    var vectorizer: Vectorizer
        get() = model.vectorizer
        set(value) {
            clusterer.vectorizer = value
            model.vectorizer = value
        }

    var pagenum: Int
        get() = model.pagenum
        set(value) {
            model.pagenum = value
        }

    var path: String = ""
        get() = field
        set(value) {
            field = value
            model.path = value
            val document = PDDocument.load(File(value))
            view.fieldFilepath.text = value
            view.comboboxPagenum.removeAllItems()
            (0..document.numberOfPages).forEach(view.comboboxPagenum::addItem)
            document.close()
        }

    private fun validateCluster(): ValidationError {
        return when {
            !Files.exists(Paths.get(model.path)) -> ValidationError.INVALIDPATH
            else -> ValidationError.NONE
        }
    }

    private fun validateMerge(): ValidationError {
        return when {
            model.threshold < 0 -> ValidationError.INVALIDTHRESHOLD
            !Files.exists(Paths.get(model.path)) -> ValidationError.INVALIDPATH
            model.clusters == null -> ValidationError.NOTCLUSTERED
            else -> ValidationError.NONE
        }
    }

    fun cluster() {
        val error = validateCluster()

        if (error == ValidationError.NONE) {
            thread(start = true) {
                view.btnCluster.isEnabled = false
                view.labelStatus.text = "Clustering..."
                val document = PDDocument.load(File(model.path))
                model.clusters = clusterer.clusterFilePage(document, model.pagenum)
                view.btnCluster.isEnabled = true
                view.labelStatus.text = "Clustering finished"
                view.frame.repaint()
                view.frame.revalidate()
                document.close()
            }
        } else {
            view.labelStatus.text = error.toString()
        }
    }

    fun merge() {
        val error = validateMerge()

        if (error == ValidationError.NONE) {
            thread(start = true) {
                view.btnMerge.isEnabled = false
                view.labelStatus.text = "Merging..."
                val document = PDDocument.load(File(model.path))
                model.clusters?.let { clusters ->
                    val merged = collectBelowCutoff(clusters, model.threshold)
                    merged.map(clusterer::getBoundingRect).forEach {
                        drawRect(document, document.getPage(model.pagenum), it)
                    }
                    val img = PDFRenderer(document).renderImage(model.pagenum)
                    view.labelPdfViewer.icon = ImageIcon(img)
                    view.btnMerge.isEnabled = true
                    view.labelStatus.text = "Merging finished"
                    view.labelPdfViewer.isVisible = true
                    view.frame.repaint()
                    view.frame.revalidate()
                    view.frame.pack()
                }
                document.close()
            }
        } else {
            view.labelStatus.text = error.toString()
        }
    }

    enum class ValidationError {
        NONE,
        INVALIDTHRESHOLD {
            override fun toString() = "Threshold should be a number >= 0"
        },
        INVALIDPATH {
            override fun toString() = "Given path does not exist"
        },
        NOTCLUSTERED {
            override fun toString() = "Need to cluster before merging"
        }
    }
}
*/
