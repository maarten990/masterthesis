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


fun clusterFilePage(document: PDDocument, threshold: Int, pagenum: Int) {
    val parser = TextRectParser()
    val clusterer = Clusterer()

    val chars = parser.getCharsOnPage(document, pagenum)

    var clusters = clusterer.cluster(chars)
    var mergedClusters = collectBelowCutoff(clusters, threshold)
    println("${mergedClusters.size} word-clusters")

    clusters = clusterer.recluster(mergedClusters)
    mergedClusters = collectBelowCutoff(clusters, 2 * threshold)
    println("${mergedClusters.size} block-clusters")

    val page = document.getPage(pagenum)
    mergedClusters
            .map(clusterer::getBoundingRect)
            .forEach { drawRect(document, page, it) }
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
