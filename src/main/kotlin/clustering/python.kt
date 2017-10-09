package clustering

import gui.Vectorizer
import java.io.File
import java.io.FileReader
import com.opencsv.CSVReader
import java.nio.file.Files
import java.nio.file.Paths


abstract class Dendrogram {
    abstract fun leafNodes(): List<LeafNode>
};

class LeafNode(val data: CharData): Dendrogram() {
    override fun leafNodes(): List<LeafNode> = listOf(this)
}

class MergeNode(val left: Dendrogram, val right: Dendrogram, val dist: Double): Dendrogram() {
    override fun leafNodes(): List<LeafNode> = listOf(left, right).flatMap(Dendrogram::leafNodes)
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