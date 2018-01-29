package clustering

sealed class Dendrogram {
    abstract fun leafNodes(): List<LeafNode>
    abstract fun childDistances(): List<Double>

    fun collectBelowCutoff(cutoff: Int): Map<CharData, Int> {
        val blocks = collectBlocksBelowCutoff(cutoff)
        val indices = blocks.indices
        val out = mutableMapOf<CharData, Int>()

        for ((i, cluster) in indices.zip(blocks)) {
            cluster.forEach { out[it] = i }
        }

        return out
    }

    private fun collectBlocksBelowCutoff(cutoff: Int): List<List<CharData>> {
        return if (this is LeafNode) {
            listOf(listOf(this.data))
        } else if (this is MergeNode && dist <= cutoff) {
            listOf(leafNodes().map(LeafNode::data))
        } else {
            this as MergeNode
            listOf(left, right).flatMap { it.collectBlocksBelowCutoff(cutoff)}
        }
    }

    fun collectDistances(cutoff: Int): List<Double> {
         return if (this is LeafNode) {
             listOf()
        } else if (this is MergeNode && dist <= cutoff) {
            childDistances()
        } else {
             this as MergeNode
             left.collectDistances(cutoff) + right.collectDistances(cutoff)
        }
    }

    companion object Factory {
        fun fromLists(data: List<CharData>, clusters: List<List<Double>>): Dendrogram {
            val nodes: MutableList<Dendrogram> = data.map(::LeafNode).toMutableList()

            for ((left, right, dist, _) in clusters) {
                nodes.add(MergeNode(nodes[left.toInt()], nodes[right.toInt()], dist))
            }

            if (nodes.isNotEmpty())
                return nodes.last()
            else
                return LeafNode(CharData(0.0f, 0.0f, 0.0f, 0.0f, "", 0.0f, 0.0f))
        }
    }
}

class LeafNode(val data: CharData): Dendrogram() {
    override fun leafNodes() = listOf(this)
    override fun childDistances() = listOf<Double>()
}

class MergeNode(val left: Dendrogram, val right: Dendrogram, val dist: Double): Dendrogram() {
    override fun leafNodes() = listOf(left, right).flatMap(Dendrogram::leafNodes)
    override fun childDistances() = listOf(dist) + left.childDistances() + right.childDistances()
}
