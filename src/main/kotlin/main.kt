package main

import com.apporiented.algorithm.clustering.Cluster
import com.apporiented.algorithm.clustering.DefaultClusteringAlgorithm
import com.apporiented.algorithm.clustering.SingleLinkageStrategy
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.PDPageContentStream
import java.awt.Color
import java.io.File
import java.lang.Math.pow
import java.lang.Math.sqrt


fun main(args: Array<String>) {
    val f = File(args[0])

    val parser = TextRectParser()
    for (cutoff in listOf(1, 5, 7, 10, 25, 50)) {
        val doc = PDDocument.load(f)
        for (pageNum in 2..2) {
            val chars = parser.getCharsOnPage(doc, pageNum)
            val clusters = clusterChars(chars.values)

            val words = collectBelowCutoff(clusters, cutoff)
            println("${words.size} word-clusters")

            val page = doc.getPage(pageNum)
            words
                    .map { word -> word.map { chars[it.toInt()] }.requireNoNulls() }
                    .map { getBoundingRect(it) }
                    .forEach { drawRect(doc, page, it) }
        }

        doc.save("modified-$cutoff.pdf")
        doc.close()
    }
}


fun getBoundingRect(chars: List<CharData>): CharData {
    val leftMost = chars.map(CharData::left).min()!!
    val rightMost = chars.map { it.left + it.width }.max()!!
    val topMost = chars.map(CharData::top).min()!!
    val botMost = chars.map { it.top + it.height }.max()!!

    return CharData(leftMost, topMost, rightMost - leftMost, botMost - topMost,
            "0", 0.0f, 0.0f)
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


fun getDistanceMatrix(chars: Collection<CharData>, metric: (CharData, CharData) -> Double): Array<DoubleArray> {
    return chars
            .map {c1 -> chars.map { c2 -> metric(c1, c2) }.toDoubleArray() }
            .toTypedArray()
}


fun clusterChars(chars: Collection<CharData>): Cluster {
    val matrix = getDistanceMatrix(chars, ::euclidian)
    val clusterer = DefaultClusteringAlgorithm()

    return clusterer.performClustering(matrix,
            chars.map { it.name }.toTypedArray(), SingleLinkageStrategy())
}


fun collectBelowCutoff(cluster: Cluster, cutoff: Int) : Collection<Set<String>> {
    return if (cluster.distanceValue <= cutoff || cluster.isLeaf) {
        listOf(cluster.getLeafs().map { it.name!! }.toSet())
    } else {
        cluster.children.flatMap { collectBelowCutoff(it, cutoff) }
    }
}


fun Cluster.getLeafs(): Collection<Cluster> {
    if (this.isLeaf) {
        return listOf(this)
    } else {
        return this.children.flatMap(Cluster::getLeafs)
    }
}