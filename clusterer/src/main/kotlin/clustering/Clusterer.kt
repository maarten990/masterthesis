package clustering

import gui.Vectorizer
import org.apache.pdfbox.pdmodel.PDDocument


class Clusterer {
    var vectorizer: Vectorizer = Vectorizer.ALL
    private val python = PythonEnv()

    /**
     * Run kmeans clustering on the distances in the dendrogram, using a given cutoff point.
     */
    fun clusterDistances(tree: Dendrogram, cutoff: Float): List<Double> {
        return python.cluster_distances(tree, cutoff)
    }

    /**
     * Hierarchically cluster a collection of items.
     */
    fun cluster(chars: List<CharData>): Dendrogram {
        return python.cluster(chars, vectorizer)
    }

    /**
     * Cluster a collection of items using DBSCAN.
     */
    fun dbscan(chars: List<CharData>, epsilon: Float, minSamples: Int): Map<CharData, List<Double>> {
        return python.dbscan(chars, vectorizer, epsilon, minSamples)
    }

    /**
     * Cluster a collection of items using KMeans.
     */
    fun kmeans(chars: List<CharData>, k: Int): Map<CharData, List<Double>> {
        return python.kmeans(chars, vectorizer, k)
    }

    /**
     * Cluster a collection of items using a Gaussian mixture model.
     */
    fun gmm(chars: List<CharData>, k: Int): Map<CharData, List<Double>> {
        return python.gmm(chars, vectorizer, k)
    }

    /**
     * Recluster hierarchically based on the bounding rectangles of each cluster.
     */
    fun recluster(clusters: Collection<List<CharData>>): Dendrogram {
        val bboxes = clusters.map(::getBoundingRect)
        return cluster(bboxes)
    }

    /**
     * Hierarchically cluster the text on a PDF page.
     */
    fun clusterFilePage(document: PDDocument, pagenum: Int, filename: String): Dendrogram {
        val parser = TextRectParser()
        val chars = parser.getCharsOnPage(document, pagenum, filename)
        return cluster(chars)
    }

    /**
     * Cluster the text on a PDF page using DBSCAN.
     */
    fun clusterFilePageDbscan(document: PDDocument, pagenum: Int, filename: String,
                              epsilon: Float, minSamples: Int, use_gmm: Int = 0): Map<CharData, Int> {
        val parser = TextRectParser()
        val chars = parser.getCharsOnPage(document, pagenum, filename)

        return if (use_gmm > 0) {
            gmm(chars, use_gmm).mapValues { argmax(it.value) }
        } else {
            dbscan(chars, epsilon, minSamples).mapValues { argmax(it.value) }
        }
    }
}

fun <T: Comparable<T>>argmax(xs: List<T>): Int {
    return (0 until xs.size).maxBy { xs[it] }!!
}

/**
 * Return the bounding rectangle of a collection of nodes.
 */
fun getBoundingRect(chars: Collection<CharData>): CharData {
    // get the bounding rectangles for each clusters and recluster based on them
    val leftMost = chars.map(CharData::left).min() ?: 0.0f
    val rightMost = chars.map { it.left + it.width }.max() ?: 0.0f
    val botMost = chars.map { it.bottom }.min() ?: 0.0f
    val topMost = chars.map{ it.bottom + it.height }.max() ?: 0.0f

    val fontSize = getMode(chars.map(CharData::fontSize)) ?: 0.0f
    val fontID = getMode(chars.map(CharData::fontID)) ?: 0.0f
    val page = getMode(chars.map(CharData::page)) ?: 0
    val filename = getMode(chars.map(CharData::file)) ?: ""
    val clusterText = chars
            // group clusters on the same line
            .groupBy { it.bottom }
            .merge({ a, b -> Math.abs(a - b) < 3 }, { a, b -> a + b })
            .entries
            // sort them from left to right and join the whole thing into a string
            .sortedByDescending { it.key }
            .map { it.value }
            .map { it.sortedBy(CharData::left) }
            .joinToString("\n", transform={ it.joinToString("", transform=CharData::ch) })

    return CharData(leftMost, botMost, rightMost - leftMost, topMost - botMost,
            clusterText, fontSize, fontID, page, filename)
}


/**
 * Return the mode (most common element) of a list.
 */
fun<T> getMode(items: Collection<T>): T? {
    return items
            .groupBy { it }
            .entries
            .maxBy { it.value.size }
            ?.key
}

/**
 * Merge items that are close enough together to be essentially equal.
 *
 * @pred: Predicate that checks keys for equality.
 * @mergeFunc: Function that merges two values.
 */
fun<V> Map<Float, V>.merge(pred: (Float, Float) -> Boolean, mergeFunc: (V, V) -> V): Map<Float, V> {
    val s = this.entries.sortedBy { it.key }.map { Pair(it.key, it.value) }.toMutableList()
    var done = false

    // TODO: maybe not use basically bubblesort
    while (done) {
        var changed = false
        for (i in 0 until s.size - 1) {
            if (pred(s[i].first, s[i + 1].first)) {
                s[i] = Pair(s[i].first, mergeFunc(s[i].second, s[i + 1].second))
                s.drop(i + 1)
                changed = true
            }
        }

        done = !changed
    }

    return s.toMap()
}
