package main

import com.apporiented.algorithm.clustering.Cluster
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.PDPageContentStream
import java.awt.Color
import java.io.File


fun main(args: Array<String>) {
    val f = File(args[0])

    val parser = TextRectParser()
    val clusterer = Clusterer()
    for (cutoff in listOf(1, 5, 7, 10)) {
        val doc = PDDocument.load(f)
        for (pageNum in 2..2) {
            val chars = parser.getCharsOnPage(doc, pageNum)
            val clusters = clusterer.cluster(chars)

            val words = collectBelowCutoff(clusters, cutoff)
            println("${words.size} word-clusters")

            val page = doc.getPage(pageNum)
            for (wordCluster in words) {
                val bbox = getBoundingRect(wordCluster.map {
                    clusterer.lookupTable[it]!!
                })
                drawRect(doc, page, bbox)
            }
        }

        doc.save("modified-$cutoff.pdf")
        doc.close()
    }
}


fun getBoundingRect(chars: List<CharData>): CharData {
    val leftMost = chars.map(CharData::left).min() ?: 0.0f
    val rightMost = chars.map { it.left + it.width }.max() ?: 0.0f
    val topMost = chars.map(CharData::top).min() ?: 0.0f
    val botMost = chars.map { it.top + it.height }.max() ?: 0.0f

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