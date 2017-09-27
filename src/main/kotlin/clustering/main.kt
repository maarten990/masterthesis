package clustering

import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.PDPageContentStream
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

fun collectBelowCutoff(cluster: Dendrogram, cutoff: Int): List<List<LeafNode>> {
    return if (cluster is LeafNode) {
        listOf(listOf(cluster))
    } else if (cluster is MergeNode && cluster.dist <= cutoff) {
        listOf(cluster.leafNodes().toList())
    } else {
        cluster as MergeNode
        listOf(cluster.left, cluster.right).flatMap { collectBelowCutoff(it, cutoff)}
    }
}

fun collectAtLevel(cluster: Dendrogram, level: Int, currentLevel: Int=0): List<List<LeafNode>> {
    return if (currentLevel >= level) {
        listOf(cluster.leafNodes())
    } else {
        if (cluster is MergeNode)
            listOf(cluster.left, cluster.right).flatMap { collectAtLevel(it, level, currentLevel + 1)}
        else
            listOf(listOf(cluster as LeafNode))
    }
}

fun collectBiggestJump(cluster: Dendrogram, offset: Int): List<List<LeafNode>> {
    return collectAtLevel(cluster, getBiggestJump(cluster, offset))
}

fun getBiggestJump(cluster: Dendrogram, offset: Int): Int {
    // get the biggest jump by going depth-first through the entire tree
    val nodes = mutableListOf(Pair(cluster, 0))
    val jumps = mutableListOf<Pair<Double, Int>>()
    while (nodes.isNotEmpty()) {
        val (node, level) = nodes.removeAt(0)
        if (node is MergeNode) {
            jumps.add(Pair(getChildDist(node) - node.dist, level))
            nodes.add(Pair(node.left, level + 1))
            nodes.add(Pair(node.right, level + 1))
        }
    }

    return jumps.sortedBy { it.first }.getOrNull(offset)?.second ?: 0
}

fun getChildDist(cluster: MergeNode): Double {
    val left = cluster.left
    val right = cluster.right

    return when {
        left is MergeNode && right is MergeNode -> (left.dist + right.dist) / 2
        left is MergeNode -> left.dist
        right is MergeNode -> right.dist
        else -> 0.0
    }
}
