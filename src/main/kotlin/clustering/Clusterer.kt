package clustering

import gui.Vectorizer
import org.apache.pdfbox.pdmodel.PDDocument


class Clusterer {
    var vectorizer: Vectorizer = Vectorizer.ALL
    private val python = PythonEnv()

    /**
     * Run kmeans clustering on the distances in the dendrogram, using a given cutoff point.
     */
    fun clusterDistances(tree: Dendrogram, cutoff: Int): List<Double> {
        return python.cluster_distances(tree, cutoff)
    }

    /**
     * Label the different kinds of clusters using K-Means clustering.
     */
    fun labelClusters(clusters: List<List<LeafNode>>, k: Int): List<Int> {
        return python.label_clusters(clusters, k)
    }

    /**
     * Hierarchically cluster a collection of items.
     */
    fun cluster(chars: List<CharData>): Dendrogram {
        return python.cluster(chars, vectorizer)
    }

    /**
     * Recluster hierarchically based on the bounding rectangles of each cluster.
     */
    fun recluster(clusters: Collection<List<LeafNode>>): Dendrogram {
        val bboxes = clusters.map(this::getBoundingRect)
        return cluster(bboxes)
    }

    /**
     * Return the bounding rectangle of a collection of nodes.
     */
    fun getBoundingRect(cluster: Collection<LeafNode>): CharData {
        // translate from the names to the actual CharData objects
        val chars = cluster.map { it.data }

        // get the bounding rectangles for each clusters and recluster based on them
        val leftMost = chars.map(CharData::left).min() ?: 0.0f
        val rightMost = chars.map { it.left + it.width }.max() ?: 0.0f
        val botMost = chars.map { it.bottom }.min() ?: 0.0f
        val topMost = chars.map{ it.bottom + it.height }.max() ?: 0.0f

        // collect all characters sequentially inside the bounding box
        val clusterText = chars.sortedBy(CharData::left).joinToString("", transform=CharData::ch)
        val fontSize = getMode(chars.map(CharData::fontSize)) ?: 0.0f
        val fontID = getMode(chars.map(CharData::fontID)) ?: 0.0f

        return CharData(leftMost, botMost, rightMost - leftMost, topMost - botMost,
                clusterText, fontSize, fontID)
    }

    /**
     * Hierarchically cluster the text on a PDF page.
     */
    fun clusterFilePage(document: PDDocument, pagenum: Int): Dendrogram {
        val parser = TextRectParser()
        val chars = parser.getCharsOnPage(document, pagenum)
        return cluster(chars)
    }
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
