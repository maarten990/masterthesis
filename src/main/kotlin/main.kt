package main

import com.apporiented.algorithm.clustering.Cluster
import com.apporiented.algorithm.clustering.DefaultClusteringAlgorithm
import com.apporiented.algorithm.clustering.SingleLinkageStrategy
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.PDPageContentStream
import org.apache.pdfbox.text.PDFTextStripper
import org.apache.pdfbox.text.TextPosition
import java.awt.Color
import java.io.File
import java.io.IOException
import java.io.ByteArrayOutputStream
import java.io.OutputStreamWriter
import java.lang.Math.pow
import java.lang.Math.sqrt

var ID = 0


/**
 * Class to hold the coordinates of a pdf character.
 */
data class CharData(val left: Float, val top: Float, val width: Float,
                    val height: Float) {
    val asVec = listOf(left, top, width, height)
    val name: String = "$ID";

    init {
        ID += 1
    }
}


/**
 * Modified textstripper which saves all the character positions in the chars
 * list.
 */
class TextRectParser() : PDFTextStripper() {
    val chars = mutableListOf<CharData>()

    init {
        super.setSortByPosition(true)
    }

    /**
     * Clear the chars list, populate it with the characters on the specified
     * page, and returns the list.
     */
    fun getCharsOnPage(doc: PDDocument, page: Int): List<CharData> {
        chars.clear()
        this.startPage = page + 1
        this.endPage = page + 1
        val dummy = OutputStreamWriter(ByteArrayOutputStream())
        writeText(doc, dummy)

        return chars
    }

    @Throws(IOException::class)
    override fun writeString(string: String, textPositions: List<TextPosition>?) {
        textPositions?.map {
            chars.add(CharData(it.xDirAdj, it.yDirAdj, it.widthDirAdj, it.heightDir))
        }
    }
}


fun main(args: Array<String>) {
    val f = File(args[0])
    val doc = PDDocument.load(f)

    val parser = TextRectParser()
    for (pageNum in 2..2) {
        val chars = parser.getCharsOnPage(doc, pageNum)
        println("Clustering")
        val cluster = clusterChars(chars)
        println("Clustered")

        cluster.children[0].

        val page = doc.getPage(pageNum)
        chars.map {char -> drawRect(doc, page, char)}
    }

    doc.save("modified.pdf")
    doc.close()
}


// Draw a char's bounding box on the specified page
fun drawRect(document: PDDocument, page: PDPage, char: CharData) {
    val pageHeight = page.trimBox.height
    val leftOffset = page.trimBox.lowerLeftX
    val topOffset = page.trimBox.lowerLeftY
    val content = PDPageContentStream(document, page, PDPageContentStream.AppendMode.APPEND, false)

    content.apply {
        addRect(char.left + leftOffset,
                (pageHeight + topOffset) - char.top,
                char.width, char.height)
        setStrokingColor(Color.RED)
        stroke()
        close()
    }
}


fun euclidian(c1: CharData, c2: CharData): Double {
    return c1.asVec
            .zip(c2.asVec)
            .map { (i, j) -> pow(i.toDouble() - j.toDouble(), 2.0) }
            .reduce(Double::plus)
            .let { sum -> sqrt(sum) }
}


fun getDistanceMatrix(chars: List<CharData>, metric: (CharData, CharData) -> Double): Array<DoubleArray> {
    return chars
            .map {c1 -> chars.map { c2 -> metric(c1, c2) }.toDoubleArray() }
            .toTypedArray()
}


fun clusterChars(chars: List<CharData>): Cluster {
    val matrix = getDistanceMatrix(chars, ::euclidian)
    val clusterer = DefaultClusteringAlgorithm()

    return clusterer.performClustering(matrix,
            chars.map {it.name}.toTypedArray(), SingleLinkageStrategy())
}