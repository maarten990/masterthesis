package main

import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.text.PDFTextStripper
import org.apache.pdfbox.text.TextPosition
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.OutputStreamWriter

var ID = 0

/**
 * Modified textstripper which saves all the character positions in the chars
 * list.
 */
class TextRectParser() : PDFTextStripper() {
    val chars = mutableMapOf<Int, CharData>()
    var fontID = 0
    var fontToID = mutableMapOf<String, Int>()


    init {
        super.setSortByPosition(true)
    }

    /**
     * Clear the chars list, populate it with the characters on the specified
     * page, and returns the list.
     */
    fun getCharsOnPage(doc: PDDocument, page: Int): Map<Int, CharData> {
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

            val data = CharData(it.xDirAdj, it.yDirAdj, it.widthDirAdj,
                    it.heightDir, it.unicode, it.fontSize, fontToID[fontStr]?.toFloat()!!)
            chars[data.id] = data
        }
    }
}

/**
 * Class to hold the coordinates of a pdf character.
 */
data class CharData(val left: Float, val top: Float, val width: Float,
                    val height: Float, val ch: String, val fontSize: Float,
                    val fontID: Float) {
    val asVec = listOf(left, top, width, height, fontSize, fontID)
    val name: String = "$ID";
    val id = ID

    init {
        ID += 1
    }
}


