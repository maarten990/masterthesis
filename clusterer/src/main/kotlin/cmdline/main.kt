package cmdline

import clustering.CharData
import clustering.Clusterer
import clustering.getBoundingRect
import gui.Vectorizer
import gui.labelMappingToLists
import javafx.application.Application
import org.apache.pdfbox.pdmodel.PDDocument
import org.docopt.Docopt
import java.io.File

val usage = """
Text block clustering.
Usage:
  clusterer gui
  clusterer <param_file> <files>...
"""

fun main(args: Array<String>) {
    val opts = Docopt(usage).withHelp(true).parse(*args)

    if (opts["gui"] == true) {
        return Application.launch(gui.ClusterApp::class.java, *args)
    }

    parseConfig(opts["<param_file>"] as String)
    /*
    val blocks = when {
        opts["dbscan"] == true -> cluster_dbscan(opts["<files>"] as List<String>, (opts["<eps>"] as String).toFloat(),
                (opts["<min_pts>"] as String).toInt())
        opts["kmeans"] == true -> listOf(mapOf())
        else -> listOf(mapOf())
    }

    val labeled = labelClusters(blocks, 3)

    for (page in blocks) {
        for (block in labelMappingToLists(page).map(::getBoundingRect)) {
            for (line in block.ch.split("\n")) {
                println("${labeled[block]}: $line")
            }
        }
    }
    */
}

fun cluster_dbscan(paths: List<String>, epsilon: Float, minSamples: Int): List<Map<CharData, Int>> {
    val blocks = mutableListOf<Map<CharData, Int>>()
    // TODO: use all files
    val doc = PDDocument.load(File(paths[0]))
    val clusterer = Clusterer()

    for (pagenum in 0 until doc.numberOfPages) {
        clusterer.vectorizer = Vectorizer.GEOM
        blocks.add(clusterer.clusterFilePageDbscan(doc, pagenum, epsilon, minSamples))
    }

    doc.close()

    return blocks
}

fun labelClusters(blocks: List<Map<CharData, Int>>, k: Int): Map<CharData, Int> {
    val clusterer = Clusterer()
    clusterer.vectorizer = Vectorizer.ONLY_DIMS

    // group the data into lists of chardata objects belonging to the same cluster, for the document as a whole
    val clusterGroups = blocks.flatMap(::labelMappingToLists)

    return clusterer.kmeans(clusterGroups.map(::getBoundingRect), k)
}
