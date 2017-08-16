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

    private fun getDistanceMatrix(chars: Collection<CharData>, metric: (CharData, CharData) -> Double): Array<DoubleArray> {
        return chars
                .map {c1 -> chars.map { c2 -> metric(c1, c2) }.toDoubleArray() }
                .toTypedArray()
    }
}

fun euclidean(c1: CharData, c2: CharData): Double {
    return c1.asVec
            .zip(c2.asVec)
            .map { (i, j) -> pow(i.toDouble() - j.toDouble(), 2.0) }
            .reduce(Double::plus)
            .let { sum -> sqrt(sum) }
}
