package gui

import org.apache.pdfbox.pdmodel.PDDocument
import org.opencompare.hac.dendrogram.DendrogramNode

data class ClusterModel(var threshold: Int, var vectorizer: Vectorizer, var pagenum: Int, var path: String,
                        var clusters: DendrogramNode?)

enum class Vectorizer {
    ALL, GEOM, CENTROID
}
