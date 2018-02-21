package cmdline

import clustering.CharData
import clustering.Clusterer
import clustering.drawRect
import clustering.getBoundingRect
import gui.Vectorizer
import gui.labelMappingToLists
import javafx.application.Application
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.docopt.Docopt
import org.w3c.dom.Element
import org.w3c.dom.Node
import org.w3c.dom.NodeList
import java.awt.Color
import java.awt.Rectangle
import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import javax.xml.transform.TransformerFactory
import javax.xml.transform.dom.DOMSource
import javax.xml.transform.stream.StreamResult

val usage = """
Text block clustering.
Usage:
  clusterer gui
  clusterer <param_file> <xml> <file>
"""

fun main(args: Array<String>) {
    val opts = Docopt(usage).withHelp(true).parse(*args)

    if (opts["gui"] == true) {
        return Application.launch(gui.ClusterApp::class.java, *args)
    }

    val conf = parseConfig(opts["<param_file>"] as String)
    val blocks = conf.clusteringFunc(opts["<file>"] as String)
    val labeled = conf.labelingFunc(blocks)

    insertIntoXml(opts["<xml>"] as String, labeled)
}

fun cluster_dbscan(path: String, epsilon: Float, minSamples: Int): List<Map<CharData, Int>> {
    val blocks = mutableListOf<Map<CharData, Int>>()
    // TODO: use all files
    val doc = PDDocument.load(File(path))
    val clusterer = Clusterer()

    for (pagenum in 0 until doc.numberOfPages) {
        clusterer.vectorizer = Vectorizer.GEOM
        blocks.add(clusterer.clusterFilePageDbscan(doc, pagenum, epsilon, minSamples))
    }

    doc.close()

    return blocks
}

fun cluster_hac(path: String, cutoff: Int): List<Map<CharData, Int>> {
    val blocks = mutableListOf<Map<CharData, Int>>()
    // TODO: use all files
    val doc = PDDocument.load(File(path))
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
    val pages = dom.getElementsByTagName("page")

    var correct = 0.0
    var total = 0.0
    var unmatched = 0

    val pdf = PDDocument.load(File("src/main/resources/18001.pdf"))

    for (page in pages.iterator()) {
        val pageNum = page.attributes.getNamedItem("number").textContent.toInt()
        val pageHeight = page.attributes.getNamedItem("height").textContent.toFloat()
        val children = page.childNodes
        val pdfPage = pdf.getPage(pageNum - 1)

        for (text in children.iterator()) {
            if (text.nodeName != "text" || !text.hasAttributes())
                continue

            val top = text.attributes.getNamedItem("top").textContent.toFloat()
            val left = text.attributes.getNamedItem("left").textContent.toFloat()
            val width = text.attributes.getNamedItem("width").textContent.toFloat()
            val height = text.attributes.getNamedItem("height").textContent.toFloat()

            val coords = xmlCoordsToPdfBox(left, top, width, height, pageHeight, pdfPage)
            // pdf.drawRect(pdfPage, CharData(coords.x.toFloat(), (coords.y - coords.height).toFloat(),
            //         coords.width.toFloat(), coords.height.toFloat(),
            //         text.textContent, 0.0f, 0.0f, pageNum))

            // for each block, check if the element's coords are within the block's
            val bestMatch = grouped[pageNum]!!.maxBy { block ->
                val blockCoords = Rectangle(block.key.left.toInt(), (block.key.bottom - block.key.height).toInt(),
                        block.key.width.toInt(), block.key.height.toInt())
                // pdf.drawRect(pdfPage, block.key, Color.GREEN)

                val intersection = blockCoords.intersection(coords)
                return@maxBy if (intersection.isEmpty) {
                    0
                } else {
                    intersection.area()
                }
            }

            (text as Element).setAttribute("clusterLabel", bestMatch?.value.toString())
            total += 1
            if (text.textContent.split(" ").all { bestMatch!!.key.ch.contains(it) }) {
                correct += 1
            }
        }

        // pdf.save(File("debug_output.pdf"))
    }

    println("Matched $total text elements, $correct correct (${(correct / total) * 100}%).")
    println("Failed to match $unmatched text elements.")

    // save the result as xml
    val transformer = TransformerFactory.newInstance().newTransformer()
    val source = DOMSource(dom)
    val result = StreamResult(File("labeled.xml"))
    transformer.transform(source, result)
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

/**
 * Convert from a top-left to a bottom-left origin and divide by 1.5 to compensate for the scale.
 */
fun xmlCoordsToPdfBox(left: Float, top: Float, width: Float, height: Float,
                      pageHeight: Float, page: PDPage): Rectangle {
    val leftT = (left / 1.5f) - page.cropBox.lowerLeftX
    val topT = ((pageHeight - top) / 1.5f) - page.cropBox.lowerLeftY
    val widthT = width / 1.5f
    val heightT = height / 1.5f

    return Rectangle(Math.round(leftT), Math.round(topT), Math.round(widthT), Math.round(heightT))
}

fun Rectangle.area(): Int {
    return width * height
}
