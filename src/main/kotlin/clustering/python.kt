package clustering

import gui.Vectorizer
import java.io.File
import java.io.FileReader
import com.opencsv.CSVReader
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Paths

class PythonEnv(private var inPath: String="in.numpy", private var outPath: String="out.numpy") {
    private var interpreterPath: String
    private var scriptFolder = "src/main/resources/"

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

    fun cluster_distances(tree: Dendrogram, cutoff: Int): List<Double> {
        saveDistances(tree, cutoff)
        return callPython("kmeans.py")[0]
    }

    fun label_clusters(clusters: List<List<LeafNode>>, k: Int): List<Int> {
        return listOf()
    }

    private fun callPython(script: String): List<List<Double>> {
        val builder = ProcessBuilder(interpreterPath, scriptFolder + script, inPath, outPath)
        val process = builder.inheritIO().start()
        process.waitFor()

        val output = loadCsv()
        cleanup()

        return output
    }

    // Ensure that the given executable is Python 3
    private fun tryExectuable(name: String): Boolean {
        val builder = ProcessBuilder(name, "-c", "\"import sys; print(sys.version_info[0])\"")

        return try {
            val process = builder.start()
            val output = process.inputStream
            process.waitFor()

            output.reader().use { it.readText() }.trim() == "3"
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
    }

    private fun saveChardata(data: List<CharData>, vectorizer: Vectorizer) {
        File(inPath).printWriter().use { out ->
            data.forEach {
                out.println(vectorizer.function(it).joinToString(","))
            }
        }
    }

    private fun saveDistances(tree: Dendrogram, cutoff: Int) {
        File(inPath).printWriter().use { out ->
            out.println(tree.collectDistances(cutoff).joinToString(","))
        }
    }

}