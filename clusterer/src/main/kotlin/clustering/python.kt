package clustering

import gui.Vectorizer
import com.opencsv.CSVReader
import java.io.*
import java.nio.file.Files
import java.nio.file.Paths

class PythonEnv {
    private var inPath = "in.numpy"
    private var outPath = "out.numpy"
    private var scriptPath = "tmp_script.py"
    private var interpreterPath: String

    init {
        interpreterPath = when {
            tryExectuable("python3") -> "python3"
            tryExectuable("python") -> "python"
            else -> throw Exception("Could not find a valid Python 3 installation in the runtime path")
        }
    }

    fun cluster(data: List<CharData>, vectorizer: Vectorizer): Dendrogram {
        saveChardata(data, vectorizer)
        val clusters = callPython("cluster.py")

        return Dendrogram.fromLists(data, clusters)
    }

    fun cluster_distances(tree: Dendrogram, cutoff: Float): List<Double> {
        saveDistances(tree, cutoff)
        return callPython("kmeans.py")[0]
    }

    fun dbscan(data: List<CharData>, vectorizer: Vectorizer, epsilon: Float, minSamples: Int): Map<CharData, List<Double>> {
        saveChardata(data, vectorizer)
        val result = callPython("dbscan.py", epsilon.toString(), minSamples.toString())

        val clusters = if (result.isNotEmpty()) {
            data.zip(result[0]).toMap().mapValues { it.value.toInt() }
        } else {
            mapOf()
        }

        val numClusters = clusters.values.distinct().size
        return clusters.mapValues { (_, v) -> onehot(v, numClusters) }
    }

    fun kmeans(data: List<CharData>, vectorizer: Vectorizer, k: Int): Map<CharData, List<Double>> {
        saveChardata(data, vectorizer)
        val result = callPython("kmeans.py", k.toString())

        val clusters = if (result.isNotEmpty()) {
            data.zip(result[0]).toMap().mapValues { it.value.toInt() }
        } else {
            mapOf()
        }

        val numClusters = clusters.values.distinct().size
        if (numClusters != k) {
            println("***WARNING: number of clusters incorrect***")
        }
        return clusters.mapValues { (_, v) -> onehot(v, numClusters) }
    }

    fun gmm(data: List<CharData>, vectorizer: Vectorizer, k: Int): Map<CharData, List<Double>> {
        saveChardata(data, vectorizer)
        val result = callPython("bgmm.py", k.toString())

        return if (result.isNotEmpty()) {
            data.zip(result).toMap()
        } else {
            mapOf()
        }
    }

    private fun callPython(script: String, vararg args: String): List<List<Double>> {
        val stream = Thread.currentThread().contextClassLoader.getResourceAsStream(script)
        val contents = stream.bufferedReader().use { it.readText() }

        File(scriptPath).printWriter().use {
            it.printf(contents)
        }

        val builder = ProcessBuilder(interpreterPath, scriptPath, inPath, outPath, *args)
        val process = builder.inheritIO().start()
        process.waitFor()

        val output = loadCsv()
        cleanup()
        return output
    }

    // Ensure that the given executable is Python 3
    private fun tryExectuable(name: String): Boolean {
        val builder = ProcessBuilder(name, "-c", "\"import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)\"")

        return try {
            builder.start().waitFor() == 0
        } catch(_: IOException) {
            false
        }
    }

    private fun loadCsv(): List<List<Double>> {
        return CSVReader(FileReader(outPath)).use { reader ->
            reader.readAll().map {
                it.map(String::toDouble)
            }
        }
    }

    private fun cleanup() {
        Files.deleteIfExists(Paths.get(inPath))
        Files.deleteIfExists(Paths.get(outPath))
        Files.deleteIfExists(Paths.get(scriptPath))
    }

    private fun saveChardata(data: List<CharData>, vectorizer: Vectorizer) {
        File(inPath).printWriter().use { out ->
            data.forEach {
                out.println(vectorizer.function(it).joinToString(","))
            }
        }
    }

    private fun saveDistances(tree: Dendrogram, cutoff: Float) {
        File(inPath).printWriter().use { out ->
            out.println(tree.collectDistances(cutoff).joinToString(","))
        }
    }

}

/**
 * Convert an index to a one-hot representation.
 */
fun onehot(i: Int, k: Int): List<Double> {
    val out = mutableListOf<Double>()
    (0 until k).forEach { out.add(0.0) }
    out[i] = 1.0
    return out
}
