package main

import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.PDPageContentStream
import org.opencompare.hac.dendrogram.DendrogramNode
import org.opencompare.hac.dendrogram.MergeNode
import org.opencompare.hac.dendrogram.ObservationNode
import java.awt.Color
import java.io.File


fun main(args: Array<String>) {
    val f = File(args[0])

    val parser = TextRectParser()
    val clusterer = Clusterer()
    val doc = PDDocument.load(f)
    val wordCutoff = 5
    val blockCutoff = 25

    for (pageNum in 2..2) {
        val chars = parser.getCharsOnPage(doc, pageNum)

        var clusters = clusterer.cluster(chars)
        var mergedClusters = collectBelowCutoff(clusters, wordCutoff)
        println("${mergedClusters.size} word-clusters")

        clusters = clusterer.recluster(mergedClusters)
        mergedClusters = collectBelowCutoff(clusters, blockCutoff)
        println("${mergedClusters.size} block-clusters")

        val page = doc.getPage(pageNum)
        mergedClusters.map(clusterer::getBoundingRect)
                .forEach { drawRect(doc, page, it) }
    }

    doc.save("modified-$wordCutoff.pdf")
    doc.close()
}


// Draw a char's bounding box on the specified page
fun drawRect(document: PDDocument, page: PDPage, char: CharData) {
    val leftOffset = page.trimBox.lowerLeftX
    val botOffset = page.trimBox.lowerLeftY
    val content = PDPageContentStream(document, page, PDPageContentStream.AppendMode.APPEND, false)

    content.apply {
        addRect(char.left + leftOffset,
                char.bottom + botOffset,
                char.width, char.height)
        setStrokingColor(Color.RED)
        stroke()
        close()
    }
}


fun collectBelowCutoff(cluster: DendrogramNode, cutoff: Int) : Collection<Set<Int>> {
    return if (cluster is ObservationNode) {
        listOf(setOf(cluster.observation))
    } else if (cluster is MergeNode && cluster.dissimilarity <= cutoff) {
        listOf(cluster.getLeafs().map { it.observation }.toSet())
    } else {
        listOf(cluster.left, cluster.right).flatMap { collectBelowCutoff(it, cutoff)}
    }
}


fun DendrogramNode.getLeafs(): Collection<ObservationNode> {
    return if (this is ObservationNode) {
        listOf(this)
    } else {
        listOf(this.left, this.right).flatMap(DendrogramNode::getLeafs)
    }
}
