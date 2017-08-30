package gui

import clustering.clusterFilePage
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.rendering.PDFRenderer
import java.awt.Color
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
            view.filepath.text = value
            view.pagenum.removeAllItems()
            model.document?.let { (0..it.numberOfPages).forEach(view.pagenum::addItem) }
        }

    fun cluster() {
        thread(start = true) {
            view.cluster_btn.isEnabled = false
            model.document?.let {
                clusterFilePage(it, model.threshold, model.pagenum)
                val img = PDFRenderer(it).renderImage(model.pagenum)
                view.pdfviewer.icon = ImageIcon(img)
                view.cluster_btn.isEnabled = true
                view.frame.repaint()
                view.frame.revalidate()
            }
        }
    }
}
