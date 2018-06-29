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
import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import javax.xml.transform.TransformerFactory
import javax.xml.transform.dom.DOMSource
import javax.xml.transform.stream.StreamResult

val usage = """
Text block clustering.
Usage:
  clusterer gui
  clusterer [-d] <param_file> <xml_folder> <pdf_folder> <out_folder>

Options:
  -d --debug  Emit a pdf with bounding box drawings.
"""

var DEBUG = false

fun main(args: Array<String>) {
    val opts = Docopt(usage).withHelp(true).parse(*args)
    if (opts["gui"] == true) {
        return Application.launch(gui.ClusterApp::class.java, *args)
    }

    val xmls = File(opts["<xml_folder>"] as String).walk()
            .filter { it.isFile }
            .map { it.path }
            .toList()
            .sorted()
    val pdfs = xmls
            .map {
                (opts["<pdf_folder>"] as String) + "/" + File(it).nameWithoutExtension + ".pdf" 
            }
            .toList()
            .sorted()
    val outPaths = File(opts["<xml_folder>"] as String).walk()
            .map {
                (opts["<out_folder>"] as String) + "/" + it.name
            }
            .toList()
            .sorted()

    DEBUG = opts["--debug"] as Boolean
    val conf = parseConfig(opts["<param_file>"] as String)
    val blocksPerFile = pdfs.map { conf.clusteringFunc(it) }

    // Flatten the list to performs the block labeling on all blocks at once, then translate it back to the list
    // structure for putting it back into the right XML files.
    conf.labelingFunc.let {
        if (it is LabelingFunc.WithCentroids) {
            val labeled = it.func(blocksPerFile.flatten())
            val (labeledPerFile, filenames) = toPages(labeled)

            for (i in 0 until xmls.size) {
                println("Writing (${xmls[i]}, ${pdfs[i]}) to ${outPaths[i]}. Using data from ${filenames[i]}")
                insertIntoXml(xmls[i], pdfs[i], outPaths[i], labeledPerFile[i])
            }
        }

        else if (it is LabelingFunc.WithDist) {
            val labeled = it.func(blocksPerFile.flatten())
            val (labeledPerFile, filenames) = toPages(labeled)

            for (i in 0 until xmls.size) {
                println("Writing (${xmls[i]}, ${pdfs[i]}) to ${outPaths[i]}. Using data from ${filenames[i]}")
                insertIntoXml(xmls[i], pdfs[i], outPaths[i], labeledPerFile[i])
            }
        }
    }
}

fun <T>toPages(labeled: Map<CharData, T>): Pair<MutableList<MutableMap<CharData, T>>, List<String>> {
    val out = mutableListOf<MutableMap<CharData, T>>()
    val files = labeled.map { (k, _) -> k.file }.distinct().toList().sorted()
    val fileNameToIdx = files.mapIndexed { index, s -> Pair(s, index) }.toMap()
    files.forEach { out.add(mutableMapOf()) }

    labeled.forEach {
        (text, label) -> out[fileNameToIdx[text.file]!!][text] = label
    }

    return Pair(out, files)
}

fun cluster_dbscan(path: String, epsilon: Float, minSamples: Int): List<Map<CharData, Int>> {
    val blocks = mutableListOf<Map<CharData, Int>>()
    // TODO: use all files
    val doc = PDDocument.load(File(path))
    val clusterer = Clusterer()

    for (pagenum in 0 until doc.numberOfPages) {
        clusterer.vectorizer = Vectorizer.GEOM
        blocks.add(clusterer.clusterFilePageDbscan(doc, pagenum, path, epsilon, minSamples))
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
        blocks.add(clusterer.clusterFilePage(doc, pagenum, path).collectBelowCutoff(cutoff))
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

fun labelGmm(blocks: List<Map<CharData, Int>>, k: Int): Map<CharData, List<Double>> {
    val clusterer = Clusterer()
    clusterer.vectorizer = Vectorizer.ONLY_DIMS

    // group the data into lists of chardata objects belonging to the same cluster, for the document as a whole
    val clusterGroups = blocks.flatMap(::labelMappingToLists)

    return clusterer.gmm(clusterGroups.map(::getBoundingRect), k)
}

fun <T>insertIntoXml(path: String, pdfPath: String, outPath: String, labels: Map<CharData, T>) {
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

    val pdf = PDDocument.load(File(pdfPath))

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
            if (DEBUG) {
                pdf.drawRect(pdfPage, CharData(coords.left.toFloat(), (coords.bottom).toFloat(),
                        coords.width.toFloat(), coords.height.toFloat(),
                        text.textContent, 0.0f, 0.0f, pageNum, ""))
            }

            // for each block, check if the element's coords are within the block's
            val bestMatch = grouped[pageNum]!!.maxBy { block ->
                val blockCoords = MyRect(block.key.left.toInt(), (block.key.bottom + block.key.height).toInt(),
                        block.key.width.toInt(), block.key.height.toInt())
                if (DEBUG) {
                    pdf.drawRect(pdfPage, block.key, Color.GREEN)
                }

                return@maxBy blockCoords.intersection(coords)
            }

            (text as Element).setAttribute("clusterLabel", bestMatch?.value.toString())
            total += 1
            if (text.textContent.split(" ").all { bestMatch!!.key.ch.contains(it) }) {
                correct += 1
            }
        }

        if (DEBUG) {
            pdf.save(File("debug_output.pdf"))
        }
    }

    System.err.println("Matched $total text elements, $correct correct (${(correct / total) * 100}%).")

    // save the result as xml
    val transformer = TransformerFactory.newInstance().newTransformer()
    val source = DOMSource(dom)
    val result = StreamResult(File(outPath))
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
                      pageHeight: Float, page: PDPage): MyRect {
    val leftT = (left / 1.5f) - page.trimBox.lowerLeftX
    val topT = ((pageHeight - top) / 1.5f) - page.trimBox.lowerLeftY
    val widthT = width / 1.5f
    val heightT = height / 1.5f

    return MyRect(Math.round(leftT), Math.round(topT), Math.round(widthT), Math.round(heightT))
}

class MyRect(val left: Int, val top: Int, val width: Int, val height: Int) {
    val right = left + width
    val bottom = top - height

    fun intersection(other: MyRect): Int {
        val x_overlap = Math.max(0, Math.min(right, other.right) - Math.max(left, other.left));
        val y_overlap = Math.max(0, Math.min(top, other.top) - Math.max(bottom, other.bottom));
        return x_overlap * y_overlap;
    }
}
