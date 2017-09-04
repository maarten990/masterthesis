package gui

import clustering.CharData
import org.opencompare.hac.dendrogram.DendrogramNode

data class ClusterModel(var threshold: Int, var vectorizer: Vectorizer, var pagenum: Int, var path: String,
                        var clusters: DendrogramNode?)

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
