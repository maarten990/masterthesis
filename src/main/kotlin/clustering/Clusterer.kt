package clustering

import gui.Vectorizer
import org.apache.pdfbox.pdmodel.PDDocument
import java.lang.Math.pow
import java.lang.Math.sqrt


class Clusterer {
    var vectorizer: Vectorizer = Vectorizer.ALL

    fun cluster(chars: List<CharData>): Dendrogram {
        return pythonCluster(chars, vectorizer)
    }

    fun recluster(clusters: Collection<List<LeafNode>>): Dendrogram {
        // get the bounding rectangles for each clusters and recluster based on them
        val bboxes = clusters.map(this::getBoundingRect)
        return cluster(bboxes)
    }

    fun getBoundingRect(cluster: Collection<LeafNode>): CharData {
        // translate from the names to the actual CharData objects
        val chars = cluster.map { it.data }

        val leftMost = chars.map(CharData::left).min() ?: 0.0f
        val rightMost = chars.map { it.left + it.width }.max() ?: 0.0f
        val botMost = chars.map { it.bottom }.min() ?: 0.0f
        val topMost = chars.map{ it.bottom + it.height }.max() ?: 0.0f

        // collect all characters sequentially inside the bounding box
        val clusterText = chars.sortedBy(CharData::left).joinToString("", transform=CharData::ch)
        return CharData(leftMost, botMost, rightMost - leftMost, topMost - botMost,
                clusterText, 0.0f, 0.0f)
    }

    fun clusterFilePage(document: PDDocument, pagenum: Int): Dendrogram {
        val parser = TextRectParser()
        val chars = parser.getCharsOnPage(document, pagenum)
        return cluster(chars)
    }
}
