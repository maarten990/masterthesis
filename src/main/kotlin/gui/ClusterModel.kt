package gui

import clustering.*
import javafx.beans.property.SimpleBooleanProperty
import javafx.beans.property.SimpleIntegerProperty
import javafx.beans.property.SimpleObjectProperty
import javafx.beans.property.SimpleStringProperty
import javafx.scene.image.Image
import org.apache.pdfbox.pdmodel.PDDocument
import tornadofx.*
import tornadofx.getValue
import tornadofx.setValue

class Params {
    val thresholdProperty = SimpleIntegerProperty(15)
    var threshold by thresholdProperty

    val vectorizerProperty = SimpleObjectProperty<Vectorizer>(Vectorizer.ALL)
    var vectorizer by vectorizerProperty

    val pagenumProperty = SimpleIntegerProperty(0)
    var pagenum by pagenumProperty

    val documentProperty = SimpleObjectProperty<PDDocument>()
    var document by documentProperty

    val pathProperty = SimpleStringProperty(null)
    var path by pathProperty

    override fun toString(): String = "$threshold, $vectorizer, $pagenum, $document, $path"
}

class ParamsModel : ItemViewModel<Params>() {
    val document = bind(Params::documentProperty)
    val pagenum = bind(Params::pagenumProperty)
    val vectorizer = bind(Params::vectorizerProperty)
    val threshold = bind(Params::thresholdProperty)
    val path = bind(Params::pathProperty)
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
}


class Status {
    val docLoadedProperty = SimpleBooleanProperty(false)
    var docLoaded by docLoadedProperty

    val runningProperty = SimpleBooleanProperty(false)
    var running by runningProperty
}

class StatusModel : ItemViewModel<Status>() {
    val docLoaded = bind(Status::docLoadedProperty)
    val running = bind(Status::runningProperty)
}


class Results {
    val imageProperty = SimpleObjectProperty<Image>()
    var image by imageProperty

    val clustersProperty = SimpleObjectProperty<Dendrogram>()
    var clusters by clustersProperty
}

class ResultsModel : ItemViewModel<Results>() {
    val image = bind(Results::imageProperty)
    val clusters = bind(Results::clustersProperty)

    val clusterer = Clusterer()
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
        override val function = ::collectBelowCutoff
        override val desc = "Threshold"
    },
    BIGGEST_JUMP {
        override val function = ::collectBiggestJump
        override val desc = "n'th biggest jump"
    };

    abstract val function: (Dendrogram, Int) -> List<List<LeafNode>>
    abstract val desc: String
}