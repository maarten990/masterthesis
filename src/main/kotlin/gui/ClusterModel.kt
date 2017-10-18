package gui

import clustering.*
import javafx.beans.property.*
import javafx.scene.image.Image
import org.apache.pdfbox.pdmodel.PDDocument
import tornadofx.*
import tornadofx.getValue
import tornadofx.setValue

class Params {
    val vectorizerProperty = SimpleObjectProperty<Vectorizer>(Vectorizer.ALL)
    var vectorizer by vectorizerProperty

    val pagenumProperty = SimpleIntegerProperty(0)
    var pagenum by pagenumProperty

    val documentProperty = SimpleObjectProperty<PDDocument>()
    var document by documentProperty

    val pathProperty = SimpleStringProperty(null)
    var path by pathProperty

    override fun toString(): String = "$vectorizer, $pagenum, $document, $path"
}

class ParamsModel : ItemViewModel<Params>() {
    val document = bind(Params::documentProperty)
    val pagenum = bind(Params::pagenumProperty)
    val vectorizer = bind(Params::vectorizerProperty)
    val path = bind(Params::pathProperty)

    init {
        item = Params()
    }
}

class MergeParams() {
    val thresholdProperty = SimpleIntegerProperty(15)
    var threshold by thresholdProperty

    val collectorProperty = SimpleObjectProperty<Collector>(Collector.THRESHOLD)
    var collector by collectorProperty
}

class MergeParamsModel : ItemViewModel<MergeParams>() {
    val threshold = bind(MergeParams::thresholdProperty)
    val collector = bind(MergeParams::collectorProperty)

    init {
        item = MergeParams()
    }
}

class StatusModel : ViewModel() {
    val docLoaded = bind{ SimpleBooleanProperty() }
    val running = bind { SimpleBooleanProperty() }
    val merged = bind { SimpleBooleanProperty() }

    init {
        docLoaded.value = false
        running.value = false
        merged.value = false
    }
}

class Results {
    val imageProperty = SimpleObjectProperty<Image>()
    var image by imageProperty

    val clustersProperty = SimpleObjectProperty<Dendrogram>()
    var clusters by clustersProperty

    val mergedProperty = SimpleListProperty<List<LeafNode>>()
    var merged by mergedProperty
}

class ResultsModel : ItemViewModel<Results>() {
    val image = bind(Results::imageProperty)
    val clusters = bind(Results::clustersProperty)
    val merged = bind(Results::mergedProperty)

    val clusterer = Clusterer()

    init {
        item = Results()
    }
}

enum class Vectorizer {
    ALL {
        override fun function(data: CharData) = data.asVec
    },
    GEOM {
        override fun function(data: CharData) = data.asGeomVec
    },
    CENTROID {
        override fun function(data: CharData) = data.asCentroidVec
    };

    abstract fun function(data: CharData): List<Float>
}

enum class Collector {
    THRESHOLD {
        override val function = Dendrogram::collectBelowCutoff
        override val desc = "Threshold"
    };

    abstract val function: (Dendrogram, Int) -> List<List<LeafNode>>
    abstract val desc: String
}