package gui

import javafx.collections.ObservableList
import javafx.scene.image.ImageView
import javafx.stage.FileChooser
import org.apache.pdfbox.pdmodel.PDDocument
import tornadofx.*
import java.io.File


class ClusterApp: App(ClusterView::class)

class ClusterView: View() {
    val mergePane = MergePane()
    var imView: ImageView by singleAssign()

    val vectorizeOptions = Vectorizer.values().toList().observable()
    var pageNums: ObservableList<Int> = mutableListOf<Int>().observable()
    val param: ParamsModel by inject()
    val status: StatusModel by inject()
    val results: ResultsModel by inject()

    val controller: ClusterController by inject()

    override val root = borderpane {
        param.validate(decorateErrors = false)
        left = vbox {
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

                    button("Recluster") {
                        enableWhen { status.merged }
                        action {
                            param.commit()
                            controller.recluster()
                        }
                    }
                }
            }

            this.add(mergePane.root)

            button("Print centroids") {
                action {
                    controller.kmeans()
                }
            }
        }

        bottom = progressbar(-1.0) {
                useMaxWidth = true
                visibleWhen { status.running }
        }

        center = vbox {
            hbox {
                button("Zoom in") {
                    action {
                        imView.scaleX += 0.25
                        imView.scaleY += 0.25
                    }
                }

                button("Zoom out") {
                    action {
                        imView.scaleX -= 0.25
                        imView.scaleY -= 0.25
                    }
                }
            }
            scrollpane {
                imView = imageview(results.image)
            }
        }
    }
}

class MergePane: View() {
    val mergeParam: MergeParamsModel by inject()
    val collectOptions = Collector.values().toList().observable()
    var paramLabel: Field by singleAssign()
    val controller: ClusterController by inject()

    override val root = form {
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

