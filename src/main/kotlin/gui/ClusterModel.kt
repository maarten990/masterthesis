package gui

import clustering.*
import javafx.beans.property.*
import javafx.scene.image.Image
import org.apache.pdfbox.pdmodel.PDDocument
import tornadofx.*
import tornadofx.getValue
import tornadofx.setValue
import java.awt.Color

class ProgramState {
    // the blocks of text found on a page, per page
    val blocksProperty = SimpleListProperty<Map<CharData, Int>>()
    var blocks by blocksProperty

    // a dendrogram clustering all the chars on a page
    val dendrogramProperty = SimpleListProperty<Dendrogram>()
    var dendrogram by dendrogramProperty

    // the image to display
    val imageProperty = SimpleObjectProperty<Image>()
    var image by imageProperty

    // the collector to use for cutting the dendrogram
    val collectorProperty = SimpleObjectProperty<Collector>(Collector.THRESHOLD)
    var collector by collectorProperty

    // the threshold to pass to the collector
    val thresholdProperty = SimpleIntegerProperty(15)
    var threshold by thresholdProperty

    // the PDF's path on disk
    val pathProperty = SimpleStringProperty()
    var path by pathProperty

    // the loaded pdf
    val documentProperty = SimpleObjectProperty<PDDocument>()
    var document by documentProperty

    // the page to display
    val pagenumProperty = SimpleIntegerProperty(0)
    var pagenum by pagenumProperty

    // the vectorizer to use on CharData objects
    val vectorizerProperty = SimpleObjectProperty<Vectorizer>(Vectorizer.GEOM)
    var vectorizer by vectorizerProperty

    // epsilon parameter for dbscan
    val epsilonProperty = SimpleFloatProperty(16.0f)
    var epsilon by epsilonProperty

    // min_samples parameter for dbscan
    val minSamplesProperty = SimpleIntegerProperty(1)
    var minSamples by minSamplesProperty

    // k for kmeans block labeling
    val kProperty = SimpleIntegerProperty(3)
    var k by kProperty

    // vectorizer for block labeling
    val kVectProperty = SimpleObjectProperty<Vectorizer>(Vectorizer.ONLY_DIMS)
    var kVect by kVectProperty

    // block labeling method
    val labelerProperty = SimpleObjectProperty<BlockLabeler>(BlockLabeler.KMEANS)
    val labeler by labelerProperty

    // mapping of blocks to their clustercolor
    val colormapProperty = SimpleMapProperty<CharData, Color>()
    val colormap by colormapProperty
}

class StateModel : ItemViewModel<ProgramState>() {
    val blocks = bind(ProgramState::blocksProperty)
    val dendrogram = bind(ProgramState::dendrogramProperty)
    val image = bind(ProgramState::imageProperty)
    val collector = bind(ProgramState::collectorProperty)
    val threshold = bind(ProgramState::thresholdProperty)
    val path = bind(ProgramState::pathProperty)
    val document = bind(ProgramState::documentProperty)
    val pagenum = bind(ProgramState::pagenumProperty)
    val vectorizer = bind(ProgramState::vectorizerProperty)
    val epsilon = bind(ProgramState::epsilonProperty)
    val minSamples = bind(ProgramState::minSamplesProperty)
    val k = bind(ProgramState::kProperty)
    val kVect = bind(ProgramState::kVectProperty)
    val labeler = bind(ProgramState::labelerProperty)
    val colormap = bind(ProgramState::colormapProperty)

    // program status
    val docLoaded = bind { SimpleBooleanProperty() }
    val progress = bind { SimpleFloatProperty() }
    val merged = bind { SimpleBooleanProperty() }

    init {
        item = ProgramState()

        docLoaded.value = false
        progress.value = 0.0f
        merged.value = false
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
    },
    ONLY_DIMS {
        override fun function(data: CharData) = data.asDims
    };

    abstract fun function(data: CharData): List<Float>
}

enum class Collector {
    THRESHOLD {
        override val function = Dendrogram::collectBelowCutoff
        override val desc = "Threshold"
    };

    abstract val function: (Dendrogram, Int) -> Map<CharData, Int>
    abstract val desc: String
}

enum class BlockLabeler {
    KMEANS,
    DBSCAN;
}