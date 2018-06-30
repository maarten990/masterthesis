package clustering

import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.PDPageContentStream
import org.apache.pdfbox.text.PDFTextStripper
import org.apache.pdfbox.text.TextPosition
import java.awt.Color
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.OutputStreamWriter
import java.io.Serializable

/**
 * Modified textstripper which saves all the character positions in the chars
 * list.
 */
class TextRectParser : PDFTextStripper() {
    private val chars = mutableListOf<CharData>()
    private var fontID = 0
    private var fontToID = mutableMapOf<String, Int>()
    private var currentPage = 0
    private var filename = ""


    init {
        super.setSortByPosition(true)
    }

    /**
     * Clear the chars list, populate it with the characters on the specified
     * page, and returns the list.
     */
    fun getCharsOnPage(doc: PDDocument, page: Int, filename: String): List<CharData> {
        chars.clear()
        fontToID.clear()
        this.startPage = page + 1
        this.endPage = page + 1
        currentPage = page + 1
        this.filename = filename
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
                    it.heightDir, it.unicode, it.fontSize, fontToID[fontStr]?.toFloat()!!, currentPage, filename)
            chars.add(data)
        }
    }
}

/**
 * Class to hold the coordinates of a pdf character.
 */
data class CharData(var left: Float, var bottom: Float, var width: Float,
               var height: Float, var ch: String, var fontSize: Float,
               var fontID: Float, var page: Int, var file: String) {
    constructor(str: String) : this(0.0f, 0.0f, 0.0f, 0.0f, "", 0.0f, 0.0f, 0, "") {
        val args = str.split("☝")
        this.left = args[0].toFloat()
        this.bottom = args[1].toFloat()
        this.width = args[2].toFloat()
        this.height = args[3].toFloat()
        this.ch = args[4]
        this.fontSize = args[5].toFloat()
        this.fontID = args[6].toFloat()
        this.page = args[7].toInt()
        this.file = args[8]
    }

    override fun toString(): String {
        return listOf(left, bottom, width, height, ch, fontSize, fontID, page, file).joinToString("☝") { it.toString() }
    }
}

fun CharData.asVec(): List<Float> = listOf(left, bottom, width, height, fontSize, fontID)
fun CharData.vecLabels(): List<String> = listOf("left", "bottom", "width", "height", "fontsize", "fontID")

fun CharData.asGeomVec(): List<Float> = listOf(left, left + width, bottom, bottom + height, fontSize, fontID)
fun CharData.geomLabels(): List<String> = listOf("left", "right", "bottom", "top", "fontsize", "fontID")

fun CharData.asCentroidVec(): List<Float> = listOf(left + (0.5f * width), bottom + (0.5f * height), fontSize, fontID)
fun CharData.centroidLabels(): List<String> = listOf("x_coord", "y_coord", "fontsize", "fontID")

fun CharData.asDims(): List<Float> = listOf(width, height, fontSize, fontID)
fun CharData.dimsLabels(): List<String> = listOf("width", "height", "fontsize", "fontID")

// extend TextPosition to get the y coordinate relative to a bottom-left origin
fun TextPosition.yBottom(): Float {
    /*
    y_top = height - y_bot
    -y_bot = y_top - height
    y_bot = -y_top + height
    y_bot = height - y_top
    */
    return pageHeight - y
}

// Draw a char's bounding box on the specified page
fun PDDocument.drawRect(page: PDPage, char: CharData, color: Color=Color.RED) {
    val leftOffset = page.trimBox.lowerLeftX
    val botOffset = page.trimBox.lowerLeftY
    val content = PDPageContentStream(this, page, PDPageContentStream.AppendMode.APPEND, false)

    content.apply {
        addRect(char.left + leftOffset,
                char.bottom + botOffset,
                char.width, char.height)
        setStrokingColor(color)
        stroke()
        close()
    }
}
