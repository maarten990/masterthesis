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
import javax.xml.parsers.DocumentBuilderFactory

val usage = """
Text block clustering.
Usage:
  clusterer gui
  clusterer <param_file> <xml> <files>...
"""

fun main(args: Array<String>) {
    val opts = Docopt(usage).withHelp(true).parse(*args)

    if (opts["gui"] == true) {
        return Application.launch(gui.ClusterApp::class.java, *args)
    }

    val conf = parseConfig(opts["<param_file>"] as String)
    val blocks = conf.clusteringFunc(opts["<files>"] as List<String>)
    val labeled = conf.labelingFunc(blocks)

    insertIntoXml(opts["<xml>"] as String, labeled)
    return

    for (page in blocks) {
        for (block in labelMappingToLists(page).map(::getBoundingRect)) {
            for (line in block.ch.split("\n")) {
                println("${labeled[block]}: $line")
            }
        }
    }
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

fun cluster_hac(paths: List<String>, cutoff: Int): List<Map<CharData, Int>> {
    val blocks = mutableListOf<Map<CharData, Int>>()
    // TODO: use all files
    val doc = PDDocument.load(File(paths[0]))
    val clusterer = Clusterer()

    for (pagenum in 0 until doc.numberOfPages) {
        clusterer.vectorizer = Vectorizer.GEOM
        blocks.add(clusterer.clusterFilePage(doc, pagenum).collectBelowCutoff(cutoff))
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

fun labelDbscan(blocks: List<Map<CharData, Int>>, eps: Float, min_pts: Int): Map<CharData, Int> {
    val clusterer = Clusterer()
    clusterer.vectorizer = Vectorizer.ONLY_DIMS

    // group the data into lists of chardata objects belonging to the same cluster, for the document as a whole
    val clusterGroups = blocks.flatMap(::labelMappingToLists)

    return clusterer.dbscan(clusterGroups.map(::getBoundingRect), eps, min_pts)
}

fun insertIntoXml(path: String, labels: Map<CharData, Int>) {
    val parser = DocumentBuilderFactory.newInstance().newDocumentBuilder()
    val dom = parser.parse(File(path))
    val elements = dom.getElementsByTagName("text")

    (0 until elements.length)
            .map { elements.item(it) }
            .forEach { println("${it.attributes.getNamedItem("top")}: ${it.textContent}") }
}

