package clustering

import gui.Vectorizer
import java.io.File
import java.io.FileReader
import com.opencsv.CSVReader


fun saveChardata(data: List<CharData>, vectorizer: Vectorizer, path: String) {
    File(path).printWriter().use { out ->
        data.forEach {
            out.println(vectorizer.function(it).joinToString(","))
        }
    }
}


fun callPython(inPath: String, outPath: String) {
    val cmd = "python src/main/resources/cluster.py $inPath $outPath"
    Runtime.getRuntime().exec(cmd)
}


fun loadCsv(path: String): List<List<Double>> {
    val reader = CSVReader(FileReader(path))

    return reader.readAll().map {
        it.map(String::toDouble)
    }

}


fun pythonCluster(data: List<CharData>, vectorizer: Vectorizer): List<List<Double>> {
    val inPath = "in.numpy"
    val outPath = "out.numpy"
    saveChardata(data, vectorizer, inPath)
    callPython(inPath, outPath)

    return loadCsv(outPath)
}