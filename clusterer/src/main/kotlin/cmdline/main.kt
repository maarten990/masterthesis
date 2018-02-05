package cmdline

import clustering.CharData
import clustering.Clusterer
import clustering.getBoundingRect
import gui.Vectorizer
import gui.labelMappingToLists
import javafx.application.Application
import org.apache.pdfbox.pdmodel.PDDocument
import org.docopt.Docopt
import org.w3c.dom.Node
import org.w3c.dom.NodeList
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

    /*
    for (page in blocks) {
        for (block in labelMappingToLists(page).map(::getBoundingRect)) {
            for (line in block.ch.split("\n")) {
                println("(${(block.pageHeight - (block.bottom + block.height)) * 1.5}, ${block.left * 1.5}) ${labeled[block]}: $line")
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
    // group the labels by page
    val grouped = labels.entries
            .groupBy { it.key.page }
            .mapValues { it.value.map { it.toPair()} }
            .mapValues { it.value.toMap() }

    val parser = DocumentBuilderFactory.newInstance().newDocumentBuilder()
    val dom = parser.parse(File(path))
    val pages = dom.getElementsByTagName("number")

    for (page in pages.iterator()) {
        val pageNum = page.attributes.getNamedItem("page").textContent.toInt()
        val children = page.childNodes

        for (text in children.iterator()) {
            val top = text.attributes.getNamedItem("top").textContent.toFloat()
            val left = text.attributes.getNamedItem("left").textContent.toFloat()
            val width = text.attributes.getNamedItem("width").textContent.toFloat()
            val height = text.attributes.getNamedItem("height").textContent.toFloat()

            // search all blocks for a block containing the node's coordinates
            for (block in grouped[pageNum]!!) {
                val coords = block.key.toPdfToHtmlCoords()

                // top-left origin
                if (top >= coords["top"]!!
                        && left >= coords["left"]!!
                        && top - height <= coords["bottom"]!!
                        && left + width <= coords["right"]!!) {
                    println("Found match")
                }
            }
        }
    }
}

class NodeListIterator(val list: NodeList): Iterator<Node> {
    var currentIdx = 0
    override fun next(): Node {
        currentIdx += 1
        return list.item(currentIdx - 1)
    }

    override fun hasNext() = currentIdx < list.length
}

fun NodeList.iterator(): Iterator<Node> {
    return NodeListIterator(this)
}
