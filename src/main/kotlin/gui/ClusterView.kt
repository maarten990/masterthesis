package gui

import javafx.collections.FXCollections
import javafx.scene.control.TextField
import javafx.stage.FileChooser
import tornadofx.*
import java.io.File



class ClusterApp: App(ClusterView::class)

class ClusterView: View() {
    override val root = borderpane {
        left(InputView::class)
    }
}

class InputView: View() {
    val vectorizeOptions = Vectorizer.values().toList().observable()
    var path: TextField by singleAssign()

    override val root = form {
        fieldset("Settings") {
            field("File") {
                button("Load file") {
                    action {
                        val cwd = File(System.getProperty("user.dir"))
                        val result = chooseFile("Select PDF file",
                                arrayOf(FileChooser.ExtensionFilter("PDF file", "*.pdf")),
                                op = { initialDirectory = cwd })
                        path.text = result.first().path
                    }
                }
                path = textfield()
            }

            field("Threshold") {
                textfield("15")
            }

            field("Vectorizer") {
                combobox(values = vectorizeOptions)
            }
        }
    }
}
