import net.miginfocom.swing.MigLayout
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.rendering.PDFRenderer
import java.awt.Color
import java.io.File
import javax.swing.*
import javax.swing.event.DocumentEvent
import javax.swing.event.DocumentListener
import javax.swing.filechooser.FileNameExtensionFilter
import kotlin.concurrent.thread


class ClusterView: Runnable{
    val frame = JFrame("Clusterer")
    private val controller = ClusterController(this)
    val threshold = JTextField("5")
    val filepath = JTextField("")
    val vectorizer = JComboBox<Vectorizer>(arrayOf(Vectorizer.ALL, Vectorizer.GEOM, Vectorizer.CENTROID))
    val pagenum = JComboBox<Int>()
    val cluster_btn = JButton("Cluster")
    val file_btn = JButton("Open file")
    val pdfviewer = JLabel()

    init {
        cluster_btn.addActionListener({ controller.cluster() })

        file_btn.addActionListener({
            val chooser = JFileChooser()
            chooser.fileFilter = FileNameExtensionFilter("PDF Documents", "pdf")
            chooser.currentDirectory = File(System.getProperty("user.dir"))

            if (chooser.showOpenDialog(frame) == JFileChooser.APPROVE_OPTION) {
                controller.path = chooser.selectedFile.path
            }
        })

        threshold.document.addDocumentListener(object: DocumentListener {
            override fun changedUpdate(e: DocumentEvent?) = update()
            override fun insertUpdate(e: DocumentEvent?) = update()
            override fun removeUpdate(e: DocumentEvent?) = update()
            fun update() {
                controller.threshold = threshold.text
            }
        })

        vectorizer.addActionListener({ controller.vectorizer = vectorizer.selectedItem as Vectorizer })
        pagenum.addActionListener({ controller.pagenum = pagenum.selectedItem as Int })
    }

    override fun run() {
        //Create and set up the window.
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName())
        frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE

        frame.layout = MigLayout()
        frame.add(pdfviewer, "east, grow")
        frame.add(filepath, "growx")
        frame.add(file_btn, "wrap")
        frame.add(JLabel("Page number"))
        frame.add(pagenum, "wrap")
        frame.add(JLabel("Merge threshold"))
        frame.add(threshold, "growx, wrap")
        frame.add(JLabel("Vectorization"))
        frame.add(vectorizer, "wrap")
        frame.add(cluster_btn)

        frame.pack()
        frame.isVisible = true
    }
}


class ClusterController(val view: ClusterView) {
    val model = ClusterModel(10, Vectorizer.ALL, 0, null)

    var threshold: String
        get() = model.threshold.toString()
        set(value) {
            val int = value.toIntOrNull()
            if (int == null) {
                view.cluster_btn.isEnabled = false
                view.threshold.background = Color.RED
            } else {
                model.threshold = int
                view.cluster_btn.isEnabled = true
                view.threshold.background = Color.WHITE
            }
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
            (0..model.document!!.numberOfPages).forEach(view.pagenum::addItem)
        }

    fun cluster() {
        thread(start = true) {
            view.cluster_btn.isEnabled = false
            clusterFilePage(model.document!!, model.threshold, model.pagenum)
            val img = PDFRenderer(model.document).renderImage(model.pagenum)
            view.pdfviewer.icon = ImageIcon(img)
            view.cluster_btn.isEnabled = true
            view.frame.repaint()
            view.frame.revalidate()
        }
    }
}


data class ClusterModel(var threshold: Int, var vectorizer: Vectorizer, var pagenum: Int, var document: PDDocument?)


enum class Vectorizer {
    ALL, GEOM, CENTROID
}