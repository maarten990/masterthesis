package gui

import net.miginfocom.swing.MigLayout
import java.io.File
import javax.swing.*
import javax.swing.event.DocumentEvent
import javax.swing.event.DocumentListener
import javax.swing.filechooser.FileNameExtensionFilter

class ClusterView: Runnable {
    val frame = JFrame("Clusterer")
    val controller = ClusterController(this)

    val fieldThreshold = JTextField("5")
    val fieldFilepath = JTextField("")

    val comboboxVectorizer = JComboBox<Vectorizer>(arrayOf(Vectorizer.ALL, Vectorizer.GEOM, Vectorizer.CENTROID))
    val comboboxPagenum = JComboBox<Int>()

    val btnCluster = JButton("Cluster")
    val btnFile = JButton("Open file")

    val labelPdfViewer = JLabel()
    val labelStatus = JLabel()

    init {
        btnCluster.addActionListener({ controller.cluster() })

        btnFile.addActionListener({
            val chooser = JFileChooser()
            chooser.fileFilter = FileNameExtensionFilter("PDF Documents", "pdf")
            chooser.currentDirectory = File(System.getProperty("user.dir"))

            if (chooser.showOpenDialog(frame) == JFileChooser.APPROVE_OPTION) {
                controller.path = chooser.selectedFile.path
            }
        })

        fieldThreshold.document.addDocumentListener(object: DocumentListener {
            override fun changedUpdate(e: DocumentEvent?) = update()
            override fun insertUpdate(e: DocumentEvent?) = update()
            override fun removeUpdate(e: DocumentEvent?) = update()
            fun update() {
                controller.threshold = fieldThreshold.text
            }
        })

        comboboxVectorizer.addActionListener({ controller.vectorizer = comboboxVectorizer.selectedItem as Vectorizer })
        comboboxPagenum.addActionListener({ controller.pagenum = comboboxPagenum.selectedItem as Int })
    }

    override fun run() {
        //Create and set up the window.
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName())
        frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE

        frame.layout = MigLayout()
        frame.add(labelPdfViewer, "east, grow")
        frame.add(fieldFilepath, "growx")
        frame.add(btnFile, "wrap")
        frame.add(JLabel("Page number"))
        frame.add(comboboxPagenum, "wrap")
        frame.add(JLabel("Merge fieldThreshold"))
        frame.add(fieldThreshold, "growx, wrap")
        frame.add(JLabel("Vectorization"))
        frame.add(comboboxVectorizer, "wrap")
        frame.add(btnCluster)
        frame.add(labelStatus, "growx")

        frame.pack()
        frame.isVisible = true
    }
}
