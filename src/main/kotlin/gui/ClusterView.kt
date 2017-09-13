package gui

import clustering.Clusterer
import javafx.beans.binding.Bindings
import javafx.collections.ObservableList
import javafx.stage.FileChooser
import org.apache.pdfbox.pdmodel.PDDocument
import tornadofx.*
import java.io.File


class ClusterApp: App(ClusterView::class)

class ClusterView: View() {
    val vectorizeOptions = Vectorizer.values().toList().observable()
    var pageNums: ObservableList<Int> = mutableListOf<Int>().observable()
    val param: ParamsModel by inject()
    val status: StatusModel by inject()
    val results: ResultsModel by inject()

    val controller: ClusterController by inject()

    override val root = borderpane {
        param.validate(decorateErrors = false)
        left {
            form {
                fieldset("Settings") {
                    field("File") {
                        button("Load file") {
                            action {
                                val cwd = File(System.getProperty("user.dir"))
                                val result = chooseFile("Select PDF file",
                                        arrayOf(FileChooser.ExtensionFilter("PDF file", "*.pdf")),
                                        op = { initialDirectory = cwd })

                                if (result.isEmpty()) {
                                    status.docLoaded.value = false
                                } else {
                                    param.path.value = result.first().canonicalPath
                                    param.document.value = PDDocument.load(result.first())
                                    status.docLoaded.value = param.document.value != null

                                    pageNums.clear()
                                    (0..param.document.value.numberOfPages).forEach { pageNums.add(it) }
                                }
                            }
                        }

                        label("Succesfully loaded") {
                            visibleWhen { status.docLoaded }
                        }
                    }

                    field("Page number") {
                        combobox(param.pagenum, values = pageNums) {
                            enableWhen { status.docLoaded }
                            required()
                        }
                    }

                    field("Threshold") {
                        textfield(param.threshold).required()
                    }

                    field("Vectorizer") {
                        combobox(param.vectorizer, values = vectorizeOptions).required()
                    }

                    button("Cluster") {
                        enableWhen { param.valid }
                        action {
                            controller.cluster()
                        }
                    }
                }
            }
        }

        bottom {
            progressbar(-1.0) {
                visibleWhen { status.running }
            }
        }

        center {
            scrollpane {
                imageview(results.image)
            }
        }
    }
}

