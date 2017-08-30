package clustering

import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.text.PDFTextStripper
import org.apache.pdfbox.text.TextPosition
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.OutputStreamWriter

/**
 * Modified textstripper which saves all the character positions in the chars
 * list.
 */
class TextRectParser() : PDFTextStripper() {
    private val chars = mutableListOf<CharData>()
    private var fontID = 0
    private var fontToID = mutableMapOf<String, Int>()


    init {
        super.setSortByPosition(true)
    }

    /**
     * Clear the chars list, populate it with the characters on the specified
     * page, and returns the list.
     */
    fun getCharsOnPage(doc: PDDocument, page: Int): List<CharData> {
        chars.clear()
        fontToID.clear()
        this.startPage = page + 1
        this.endPage = page + 1
        val dummy = OutputStreamWriter(ByteArrayOutputStream())
        writeText(doc, dummy)

        return chars
    }

    @Throws(IOException::class)
    override fun writeString(string: String, textPositions: List<TextPosition>?) {
        textPositions?.map {
            val fontStr = it.font.toString()
            if (fontStr !in fontToID) {
                fontToID[fontStr] = fontID
                fontID += 1
            }

            val data = CharData(it.xDirAdj, it.yBottom(), it.widthDirAdj,
                    it.heightDir, it.unicode, it.fontSize, fontToID[fontStr]?.toFloat()!!)
            chars.add(data)
        }
    }
}


/**
 * Class to hold the coordinates of a pdf character.
 */
data class CharData(val left: Float, val bottom: Float, val width: Float,
                    val height: Float, val ch: String, val fontSize: Float,
                    val fontID: Float) {
    val asVec: List<Float> = listOf(left, bottom, width, height, fontSize, fontID)
    val asGeomVec: List<Float> = listOf(left, left + width, bottom, bottom + height, fontSize, fontID)
    val asCentroidVec: List<Float> = listOf(left + (0.5f * width), bottom + (0.5f * height), fontSize, fontID)
}


// extend TextPosition to get the y coordinate relative to a bottom-left origin
fun TextPosition.yBottom(): Float {
    /*
    y_top = height - y_bot
    -y_bot = y_top - height
    y_bot = -y_top + height
    y_bot = height - y_top
    */
    return getPageHeight() - y
}


// extend TextPosition to get the private field `pageHeight`
fun TextPosition.getPageHeight(): Float {
    val field = TextPosition::class.java.getDeclaredField("pageHeight")
    field.isAccessible = true
    return field.getFloat(this)
}