package gui

import org.apache.pdfbox.pdmodel.PDDocument

data class ClusterModel(var threshold: Int, var vectorizer: Vectorizer, var pagenum: Int, var document: PDDocument?)

enum class Vectorizer {
    ALL, GEOM, CENTROID
}
