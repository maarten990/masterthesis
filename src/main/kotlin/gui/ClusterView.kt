package gui

import javafx.collections.ObservableList
import javafx.scene.control.TabPane
import javafx.scene.image.ImageView
import javafx.stage.FileChooser
import org.apache.pdfbox.pdmodel.PDDocument
import tornadofx.*
import java.io.File


class ClusterApp: App(ClusterView::class)

class ClusterView: View() {
    val paramPane = ParamTab()
    val dbscan = DBSCANTab()
    var imView: ImageView by singleAssign()

    val model: StateModel by inject()

    val vectorizeOptions = Vectorizer.values().toList().observable()
    var pageNums: ObservableList<Int> = mutableListOf<Int>().observable()

    override val root = borderpane {
        left = vbox {
            form {
                fieldset("") {
                    field("File") {
                        button("Load file") {
                            action {
                                val cwd = File(System.getProperty("user.dir"))
                                val result = chooseFile("Select PDF file",
                                        arrayOf(FileChooser.ExtensionFilter("PDF file", "*.pdf")),
                                        op = { initialDirectory = cwd })

                                if (result.isEmpty()) {
                                    model.docLoaded.value = false
                                } else {
                                    model.path.value = result.first().canonicalPath
                                    model.document.value = PDDocument.load(result.first())
                                    model.docLoaded.value = model.document.value != null

                                    pageNums.clear()
                                    (0..model.document.value.numberOfPages).forEach { pageNums.add(it) }
                                }
                            }
                        }

                        label("Succesfully loaded") {
                            visibleWhen { model.docLoaded }
                        }
                    }

                    field("Page number") {
                        combobox(model.pagenum, values = pageNums) {
                            enableWhen { model.docLoaded }
                        }
                    }

                    field("Vectorizer") {
                        combobox(model.vectorizer, values = vectorizeOptions)
                    }
                }
            }

            tabpane {
                tabClosingPolicy = TabPane.TabClosingPolicy.UNAVAILABLE
                tab("Hierarchical clustering", paramPane.root)
                tab("DBSCAN", dbscan.root)
            }
        }

        bottom = progressbar(-1.0) {
                useMaxWidth = true
                visibleWhen { model.running }
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
                imView = imageview(model.image)
            }
        }
    }
}

class ParamTab : View() {
    val mergePane = MergePane()

    val model: StateModel by inject()
    val controller: ClusterController by inject()


    override val root = vbox {
        button("Cluster") {
            action {
                model.commit()
                controller.cluster()
            }
        }

        button("Recluster") {
            enableWhen { model.merged }
            action {
                model.commit()
                controller.recluster()
            }
        }

        this.add(mergePane.root)

        button("Print centroids") {
            action {
                controller.kmeans()
            }
        }
    }
}

class MergePane: View() {
    val model: StateModel by inject()
    val collectOptions = Collector.values().toList().observable()
    var paramLabel: Field by singleAssign()
    val controller: ClusterController by inject()

    override val root = form {
        fieldset("Merge Settings") {
            field("Vectorizer") {
                combobox(model.collector, values = collectOptions) {
                    setOnAction {
                        selectedItem?.let { paramLabel.text = it.desc }
                    }
                }
            }

            paramLabel = field("Parameter") {
                textfield(model.threshold)
            }

            button("Merge") {
                action {
                    model.commit()
                    controller.merge()
                }
            }
        }
    }
}

class DBSCANTab: View() {
    val model: StateModel by inject()
    val controller: ClusterController by inject()

    override val root = form {
        fieldset("Parameters") {
            field("Epsilon") {
                textfield(model.epsilon)
            }

            field("Min samples") {
                textfield(model.minSamples)
            }

            button("Cluster") {
                action {
                    model.commit()
                    controller.cluster_dbscan()
                }
            }
        }
    }
}
