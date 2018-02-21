package cmdline

import clustering.CharData
import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import java.io.File

class DbscanParams {
    @JsonProperty
    var epsilon: Int? = null

    @JsonProperty
    var min_pts: Int? = null
}

class KmeansParams {
    var k: Int? = null
}

class HacParams {
    var cutoff: Int? = null
}

class ClusteringMethod {
    @JsonProperty
    var dbscan: DbscanParams? = null

    @JsonProperty
    var hac: HacParams? = null
}

class LabelingMethod {
    @JsonProperty
    var dbscan: DbscanParams? = null

    @JsonProperty
    var kmeans: KmeansParams? = null
}

class ConfigFormat {
    @JsonProperty
    var labeling: LabelingMethod? = null

    @JsonProperty
    var clustering: ClusteringMethod? = null
}

data class Config(var clusteringFunc: (String) -> List<Map<CharData, Int>> = { _ -> listOf(mapOf()) },
                  var labelingFunc: (List<Map<CharData, Int>>) -> Map<CharData, Int> = { _ -> (mapOf()) })

fun parseConfig(path: String): Config {
    val mapper = ObjectMapper(YAMLFactory())
    val config = mapper.readValue(File(path), ConfigFormat().javaClass)

    val conf = Config()

    config.labeling?.dbscan.let {
        if (it != null) {
            val eps = it.epsilon!!.toFloat()
            val min_pts = it.min_pts!!
            println("Labeling algorithm: dbscan, eps: $eps, min_pts: $min_pts")

            conf.labelingFunc = { files -> labelDbscan(files, eps, min_pts) }
        }
    }

    config.labeling?.kmeans.let {
        if (it != null) {
            val k = it.k!!
            println("Labeling algorithm: kmeans, k: $k")

            conf.labelingFunc = { blocks -> labelClusters(blocks, k) }
        }
    }

    config.clustering?.hac.let {
        if (it != null) {
            val cutoff = it.cutoff!!
            println("Clustering algorithm: HAC, cutoff: $cutoff")

            conf.clusteringFunc = { files -> cluster_hac(files, cutoff) }
        }
    }

    config.clustering?.dbscan.let {
        if (it != null) {
            val eps = it.epsilon!!.toFloat()
            val min_pts = it.min_pts!!
            println("Clustering algorithm: dbscan, eps: $eps, min_pts: $min_pts")

            conf.clusteringFunc = { files -> cluster_dbscan(files, eps, min_pts) }
        }
    }

    return conf
}