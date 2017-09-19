package gui

import javafx.beans.value.ChangeListener
import javafx.collections.ObservableList
import javafx.scene.control.Label
import javafx.stage.FileChooser
import org.apache.pdfbox.pdmodel.PDDocument
import tornadofx.*
import java.io.File
import javax.xml.bind.DatatypeConverter


class ClusterApp: App(ClusterView::class)

class ClusterView: View() {
    val vectorizeOptions = Vectorizer.values().toList().observable()
    val collectOptions = Collector.values().toList().observable()
    var pageNums: ObservableList<Int> = mutableListOf<Int>().observable()
    val param: ParamsModel by inject()
    val mergeParam: MergeParamsModel by inject()
    val status: StatusModel by inject()
    val results: ResultsModel by inject()

    val controller: ClusterController by inject()

    var paramLabel: Field by singleAssign()

    override val root = borderpane {
        param.validate(decorateErrors = false)
        left {
            form {
                fieldset("Cluster Settings") {
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

                    field("Vectorizer") {
                        combobox(param.vectorizer, values = vectorizeOptions).required()
                    }

                    button("Cluster") {
                        enableWhen { param.valid }
                        action {
                            param.commit()
                            controller.cluster()
                        }
                    }
                }

                fieldset("Merge Settings") {
                    field("Vectorizer") {
                        combobox(mergeParam.collector, values = collectOptions) {
                            required()
                            setOnAction {
                                selectedItem?.let { paramLabel.text = it.desc }
                            }
                        }
                    }

                    paramLabel = field("Parameter") {
                        textfield(mergeParam.threshold).validator {
                            val value = it?.toIntOrNull()
                            when {
                                value == null -> error("Could not parse integer")
                                value < 1 -> error("Value needs to be greater than zero")
                                else -> null
                            }
                        }
                    }

                    button("Merge") {
                        enableWhen { mergeParam.valid }
                        action {
                            mergeParam.commit()
                            controller.merge()
                        }
                    }
                }
            }
        }

        bottom {
            progressbar(-1.0) {
                useMaxWidth = true
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

