package main

import com.apporiented.algorithm.clustering.Cluster
import com.apporiented.algorithm.clustering.DefaultClusteringAlgorithm
import com.apporiented.algorithm.clustering.SingleLinkageStrategy
import java.lang.Math.pow
import java.lang.Math.sqrt

class Clusterer {
    var lookupTable: Map<String, CharData> = mapOf()

    fun cluster(chars: Collection<CharData>): Cluster {
        val matrix = getDistanceMatrix(chars, ::euclidean)
        val clusterer = DefaultClusteringAlgorithm()

        lookupTable = chars.map {Pair(it.hashCode().toString(), it)}.toMap()

        return clusterer.performClustering(matrix,
                chars.map { it.hashCode().toString() }.toTypedArray(),
                SingleLinkageStrategy())
    }

    fun recluster(clusters: Collection<Set<String>>): Cluster {
        // get the bounding rectangles for each clusters and recluster based on them
        val bboxes = clusters.map(this::getBoundingRect)
        return cluster(bboxes)
    }

    private fun getDistanceMatrix(chars: Collection<CharData>, metric: (CharData, CharData) -> Double): Array<DoubleArray> {
        return chars
                .map {c1 -> chars.map { c2 -> metric(c1, c2) }.toDoubleArray() }
                .toTypedArray()
    }

    fun getBoundingRect(cluster: Collection<String>): CharData {
        // translate from the names to the actual CharData objects
        val chars = cluster.map { lookupTable[it]!! }

        val leftMost = chars.map(CharData::left).min() ?: 0.0f
        val rightMost = chars.map { it.left + it.width }.max() ?: 0.0f
        val topMost = chars.map{ it.top }.max() ?: 0.0f
        val botMost = chars.map { it.top + it.height }.min() ?: 0.0f

        // collect all characters sequentially inside the bounding box
        val clusterText = chars.sortedBy(CharData::left).joinToString("", transform=CharData::ch)

        return CharData(leftMost, topMost, rightMost - leftMost, topMost - botMost,
                clusterText, 0.0f, 0.0f)
    }
}


fun euclidean(c1: CharData, c2: CharData): Double {
    return c1.asVec
            .zip(c2.asVec)
            .map { (i, j) -> pow(i.toDouble() - j.toDouble(), 2.0) }
            .reduce(Double::plus)
            .let { sum -> sqrt(sum) }
}
