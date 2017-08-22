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
    val doc = PDDocument.load(f)
    val wordCutoff = 10
    val blockCutoff = 20

    for (pageNum in 2..4) {
        val chars = parser.getCharsOnPage(doc, pageNum)

        val clusters = clusterer.cluster(chars)
        val wordNameClusters = collectBelowCutoff(clusters, wordCutoff)
        println("${wordNameClusters.size} word-clusters")

        val blockClusters = clusterer.recluster(wordNameClusters)
        val blocks = collectBelowCutoff(blockClusters, blockCutoff)
        println("${blocks.size} block-clusters")

        val page = doc.getPage(pageNum)
        blocks.map(clusterer::getBoundingRect)
                .forEach { drawRect(doc, page, it) }
    }

    doc.save("modified-$wordCutoff.pdf")
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
        setStrokingColor(Color.CYAN)
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