package clustering

import gui.ClusterView
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.PDPageContentStream
import org.opencompare.hac.dendrogram.DendrogramNode
import org.opencompare.hac.dendrogram.MergeNode
import org.opencompare.hac.dendrogram.ObservationNode
import java.awt.Color


// Draw a char's bounding box on the specified page
fun drawRect(document: PDDocument, page: PDPage, char: CharData) {
    val leftOffset = page.trimBox.lowerLeftX
    val botOffset = page.trimBox.lowerLeftY
    val content = PDPageContentStream(document, page, PDPageContentStream.AppendMode.APPEND, false)

    content.apply {
        addRect(char.left + leftOffset,
                char.bottom + botOffset,
                char.width, char.height)
        setStrokingColor(Color.RED)
        stroke()
        close()
    }
}


fun collectBelowCutoff(cluster: DendrogramNode, cutoff: Int): List<List<ObservationNode>> {
    return if (cluster is ObservationNode) {
        listOf(listOf(cluster))
    } else if (cluster is MergeNode && cluster.dissimilarity <= cutoff) {
        listOf(cluster.getLeafs().toList())
    } else {
        listOf(cluster.left, cluster.right).flatMap { collectBelowCutoff(it, cutoff)}
    }
}

fun collectAtLevel(cluster: DendrogramNode, level: Int, currentLevel: Int=0): List<List<ObservationNode>> {
    return if (currentLevel >= level) {
        listOf(cluster.getLeafs())
    } else {
        listOf(cluster.left, cluster.right).flatMap { collectAtLevel(it, level, currentLevel + 1)}
    }
}


fun collectBiggestJump(cluster: DendrogramNode): List<List<ObservationNode>> {
    return collectAtLevel(cluster, getBiggestJump(cluster))
}

fun getBiggestJump(cluster: DendrogramNode): Int {
    // get the biggest jump by going depth-first through the entire tree
    val nodes = mutableListOf(Pair(cluster, 0))
    val jumps = mutableListOf<Pair<Double, Int>>()
    while (nodes.isNotEmpty()) {
        val (node, level) = nodes.removeAt(0)
        if (node is MergeNode) {
            jumps.add(Pair(getChildDist(node) - node.dissimilarity, level))
            nodes.add(Pair(node.left, level + 1))
            nodes.add(Pair(node.right, level + 1))
        }
    }

    return jumps.maxBy { it.first }?.second ?: 0
}


fun getChildDist(cluster: MergeNode): Double {
    val left = cluster.left
    val right = cluster.right

    return when {
        left is MergeNode && right is MergeNode -> (left.dissimilarity + right.dissimilarity) / 2
        left is MergeNode -> left.dissimilarity
        right is MergeNode -> right.dissimilarity
        else -> 0.0
    }
}


fun split(clusters: Collection<Set<Int>>): Collection<List<ObservationNode>> {
    return listOf(listOf())
}


fun DendrogramNode.getLeafs(): List<ObservationNode> {
    return if (this is ObservationNode) {
        listOf(this)
    } else {
        listOf(this.left, this.right).flatMap(DendrogramNode::getLeafs)
    }
}
