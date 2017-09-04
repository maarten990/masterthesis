package clustering

import gui.ClusterView
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.PDPageContentStream
import org.opencompare.hac.dendrogram.DendrogramNode
import org.opencompare.hac.dendrogram.MergeNode
import org.opencompare.hac.dendrogram.ObservationNode
import java.awt.Color
import javax.swing.SwingUtilities


fun main(args: Array<String>) {
    val view = ClusterView()
    SwingUtilities.invokeLater(view)
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


fun collectBelowCutoff(cluster: DendrogramNode, cutoff: Int): Collection<List<ObservationNode>> {
    return if (cluster is ObservationNode) {
        listOf(listOf(cluster))
    } else if (cluster is MergeNode && cluster.dissimilarity <= cutoff) {
        listOf(cluster.getLeafs().toList())
    } else {
        listOf(cluster.left, cluster.right).flatMap { collectBelowCutoff(it, cutoff)}
    }
}


fun split(clusters: Collection<Set<Int>>): Collection<List<ObservationNode>> {
    return listOf(listOf())
}


fun DendrogramNode.getLeafs(): Collection<ObservationNode> {
    return if (this is ObservationNode) {
        listOf(this)
    } else {
        listOf(this.left, this.right).flatMap(DendrogramNode::getLeafs)
    }
}
