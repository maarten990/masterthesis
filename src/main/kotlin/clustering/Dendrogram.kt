package clustering

sealed class Dendrogram {
    abstract fun leafNodes(): List<LeafNode>

    fun collectBelowCutoff(cutoff: Int): List<List<LeafNode>> {
        return if (this is LeafNode) {
            listOf(listOf(this))
        } else if (this is MergeNode && dist <= cutoff) {
            listOf(leafNodes().toList())
        } else {
            this as MergeNode
            listOf(left, right).flatMap { it.collectBelowCutoff(cutoff)}
        }
    }

    fun collectDistances(cutoff: Int): List<Double> {
         return if (this is LeafNode) {
             listOf()
        } else if (this is MergeNode && dist <= cutoff) {
             val dists = mutableListOf<Double>()
             val stack = mutableListOf<MergeNode>(this)
             var elem: Dendrogram

             while (stack.isNotEmpty()) {
                 elem = stack.removeAt(0)
                 dists.add(elem.dist)

                 if (left is MergeNode)
                    stack.add(left)
                 if (right is MergeNode)
                     stack.add(right)
             }

             dists
        } else {
             this as MergeNode
             left.collectDistances(cutoff) + right.collectDistances(cutoff)
        }
    }
}

class LeafNode(val data: CharData): Dendrogram() {
    override fun leafNodes() = listOf(this)
}

class MergeNode(val left: Dendrogram, val right: Dendrogram, val dist: Double): Dendrogram() {
    override fun leafNodes() = listOf(left, right).flatMap(Dendrogram::leafNodes)
}
