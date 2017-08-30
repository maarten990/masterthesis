import org.opencompare.hac.HierarchicalAgglomerativeClusterer
import org.opencompare.hac.agglomeration.SingleLinkage
import org.opencompare.hac.dendrogram.DendrogramBuilder
import org.opencompare.hac.dendrogram.DendrogramNode
import org.opencompare.hac.experiment.DissimilarityMeasure
import org.opencompare.hac.experiment.Experiment
import java.lang.Math.pow
import java.lang.Math.sqrt


class Clusterer {
    var lookupTable: List<CharData> = listOf()

    fun cluster(chars: List<CharData>): DendrogramNode {
        val experiment = Experiment(chars::size)
        val measure = DissimilarityMeasure { _, i, j -> euclidean(chars[i], chars[j]) }
        val treeBuilder = DendrogramBuilder(experiment.numberOfObservations)
        val clusterer = HierarchicalAgglomerativeClusterer(experiment, measure, SingleLinkage())

        clusterer.cluster(treeBuilder)
        lookupTable = chars

        return treeBuilder.dendrogram.root
    }

    fun recluster(clusters: Collection<Set<Int>>): DendrogramNode {
        // get the bounding rectangles for each clusters and recluster based on them
        val bboxes = clusters.map(this::getBoundingRect)
        return cluster(bboxes)
    }

    fun getBoundingRect(cluster: Collection<Int>): CharData {
        // translate from the names to the actual CharData objects
        val chars = cluster.map { lookupTable[it] }

        val leftMost = chars.map(CharData::left).min() ?: 0.0f
        val rightMost = chars.map { it.left + it.width }.max() ?: 0.0f
        val botMost = chars.map { it.bottom }.min() ?: 0.0f
        val topMost = chars.map{ it.bottom + it.height }.max() ?: 0.0f

        // collect all characters sequentially inside the bounding box
        val clusterText = chars.sortedBy(CharData::left).joinToString("", transform=CharData::ch)
        return CharData(leftMost, botMost, rightMost - leftMost, topMost - botMost,
                clusterText, 0.0f, 0.0f)
    }
}


fun euclidean(c1: CharData, c2: CharData): Double {
    return c1.asGeomVec
            .zip(c2.asGeomVec)
            .map { (i, j) -> pow(i.toDouble() - j.toDouble(), 2.0) }
            .reduce(Double::plus)
            .let { sum -> sqrt(sum) }
}
