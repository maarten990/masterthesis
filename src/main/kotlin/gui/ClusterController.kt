package gui

import clustering.clusterFilePage
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.rendering.PDFRenderer
import java.io.File
import javax.swing.ImageIcon
import kotlin.concurrent.thread

class ClusterController(val view: ClusterView) {
    val model = ClusterModel(10, Vectorizer.ALL, 0, null)

    var threshold: String
        get() = model.threshold.toString()
        set(value) {
            val int = value.toIntOrNull()
            model.threshold = int ?: -1
        }

    var vectorizer: Vectorizer
        get() = model.vectorizer
        set(value) {
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
            model.document = PDDocument.load(File(value))
            view.fieldFilepath.text = value
            view.comboboxPagenum.removeAllItems()
            model.document?.let { (0..it.numberOfPages).forEach(view.comboboxPagenum::addItem) }
        }

    fun cluster() {
        thread(start = true) {
            view.btnCluster.isEnabled = false
            view.labelStatus.text = "Clustering..."
            model.document?.let {
                clusterFilePage(it, model.threshold, model.pagenum)
                val img = PDFRenderer(it).renderImage(model.pagenum)
                view.labelPdfViewer.icon = ImageIcon(img)
                view.btnCluster.isEnabled = true
                view.labelStatus.text = "Clustering finished"
                view.frame.repaint()
                view.frame.revalidate()
            }
        }
    }
}
