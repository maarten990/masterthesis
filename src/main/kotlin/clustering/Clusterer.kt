package clustering

import gui.Vectorizer
import org.apache.pdfbox.pdmodel.PDDocument
import org.opencompare.hac.HierarchicalAgglomerativeClusterer
import org.opencompare.hac.agglomeration.SingleLinkage
import org.opencompare.hac.dendrogram.DendrogramBuilder
import org.opencompare.hac.dendrogram.DendrogramNode
import org.opencompare.hac.dendrogram.ObservationNode
import org.opencompare.hac.experiment.DissimilarityMeasure
import org.opencompare.hac.experiment.Experiment
import java.lang.Math.pow
import java.lang.Math.sqrt


class Clusterer {
    var lookupTable: List<CharData> = listOf()
    var vectorizer: Vectorizer = Vectorizer.ALL

    fun cluster(chars: List<CharData>): DendrogramNode {
        val data = pythonCluster(chars, vectorizer)
        println("Python returned ${data.size} clusters")
        return ObservationNode(0)
    }

    fun recluster(clusters: Collection<List<ObservationNode>>): DendrogramNode {
        // get the bounding rectangles for each clusters and recluster based on them
        val bboxes = clusters.map(this::getBoundingRect)
        return cluster(bboxes)
    }

    fun getBoundingRect(cluster: Collection<ObservationNode>): CharData {
        // translate from the names to the actual CharData objects
        val chars = cluster.map { lookupTable[it.observation] }

        val leftMost = chars.map(CharData::left).min() ?: 0.0f
        val rightMost = chars.map { it.left + it.width }.max() ?: 0.0f
        val botMost = chars.map { it.bottom }.min() ?: 0.0f
        val topMost = chars.map{ it.bottom + it.height }.max() ?: 0.0f

        // collect all characters sequentially inside the bounding box
        val clusterText = chars.sortedBy(CharData::left).joinToString("", transform=CharData::ch)
        return CharData(leftMost, botMost, rightMost - leftMost, topMost - botMost,
                clusterText, 0.0f, 0.0f)
    }

    fun clusterFilePage(document: PDDocument, pagenum: Int): DendrogramNode {
        val parser = TextRectParser()
        val chars = parser.getCharsOnPage(document, pagenum)
        return cluster(chars)
    }
}


fun euclidean(c1: CharData, c2: CharData, vectorizer: Vectorizer): Double {
    return vectorizer.function(c1)
            .zip(vectorizer.function(c2))
            .map { (i, j) -> pow(i.toDouble() - j.toDouble(), 2.0) }
            .reduce(Double::plus)
            .let { sum -> sqrt(sum) }
}
