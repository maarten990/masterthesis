package clustering

import gui.Vectorizer
import java.io.File
import java.io.FileReader
import com.opencsv.CSVReader
import java.nio.file.Files
import java.nio.file.Paths


abstract class Dendrogram {
    abstract fun leafNodes(): List<LeafNode>

    fun collectBelowCutoff(cutoff: Int): List<List<LeafNode>> {
        return if (this is LeafNode) {
            listOf(listOf(this))
        } else if (this is MergeNode && dist <= cutoff) {
            listOf(leafNodes().toList())
        } else {
            this as MergeNode
            listOf(left, right).flatMap { it.collectBelowCutoff(cutoff)}
        }
    }

    fun collectAtLevel(level: Int, currentLevel: Int=0): List<List<LeafNode>> {
        return if (currentLevel >= level) {
            listOf(leafNodes())
        } else {
            if (this is MergeNode)
                listOf(left, right).flatMap { it.collectAtLevel(level, currentLevel + 1)}
            else
                listOf(listOf(this as LeafNode))
        }
    }

    fun collectBiggestJump(offset: Int): List<List<LeafNode>> {
        return collectAtLevel(getBiggestJump(offset))
    }

    fun getBiggestJump(offset: Int): Int {
        // get the biggest jump by going depth-first through the entire tree
        val nodes = mutableListOf(Pair(this, 0))
        val jumps = mutableListOf<Pair<Double, Int>>()
        while (nodes.isNotEmpty()) {
            val (node, level) = nodes.removeAt(0)
            if (node is MergeNode) {
                jumps.add(Pair(node.getChildDist() - node.dist, level))
                nodes.add(Pair(node.left, level + 1))
                nodes.add(Pair(node.right, level + 1))
            }
        }

        return jumps.sortedBy { it.first }.getOrNull(offset)?.second ?: 0
    }
}

class LeafNode(val data: CharData): Dendrogram() {
    override fun leafNodes(): List<LeafNode> = listOf(this)
}

class MergeNode(val left: Dendrogram, val right: Dendrogram, val dist: Double): Dendrogram() {
    override fun leafNodes(): List<LeafNode> = listOf(left, right).flatMap(Dendrogram::leafNodes)

    fun getChildDist(): Double {
        val left = left
        val right = right

        return when {
            left is MergeNode && right is MergeNode -> (left.dist + right.dist) / 2
            left is MergeNode -> left.dist
            right is MergeNode -> right.dist
            else -> 0.0
        }
    }
}

fun saveChardata(data: List<CharData>, vectorizer: Vectorizer, path: String) {
    File(path).printWriter().use { out ->
        data.forEach {
            out.println(vectorizer.function(it).joinToString(","))
        }
    }
}

fun callPython(inPath: String, outPath: String) {
    val builder = ProcessBuilder("python3", "src/main/resources/cluster.py", inPath, outPath)
    val process = builder.inheritIO().start()
    process.waitFor()
}

fun loadCsv(path: String): List<List<Double>> {
    val reader = CSVReader(FileReader(path))

    return reader.readAll().map {
        it.map(String::toDouble)
    }

}

fun createDendrogram(data: List<CharData>, clusters: List<List<Double>>): Dendrogram {
    val nodes: MutableList<Dendrogram> = data.map(::LeafNode).toMutableList()

    for ((left, right, dist, _) in clusters) {
        nodes.add(MergeNode(nodes[left.toInt()], nodes[right.toInt()], dist))
    }

    return nodes.last()
}

fun cleanup(inPath: String, outPath: String) {
    Files.deleteIfExists(Paths.get(inPath))
    Files.deleteIfExists(Paths.get(outPath))
}

fun pythonCluster(data: List<CharData>, vectorizer: Vectorizer): Dendrogram {
    val inPath = "in.numpy"
    val outPath = "out.numpy"
    saveChardata(data, vectorizer, inPath)
    callPython(inPath, outPath)
    val clusters = loadCsv(outPath)
    cleanup(inPath, outPath)

    return createDendrogram(data, clusters)
}