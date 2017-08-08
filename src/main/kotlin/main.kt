package main

import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.PDPageContentStream
import org.apache.pdfbox.text.PDFTextStripper
import org.apache.pdfbox.text.TextPosition
import java.awt.Color
import java.io.File
import java.io.IOException
import java.io.ByteArrayOutputStream
import java.io.OutputStreamWriter


/**
 * Class to hold the coordinates of a pdf character.
 */
data class CharData(val left: Float, val top: Float, val width: Float,
                    val height: Float)


/**
 * Modified textstripper which saves all the character positions in the chars
 * list.
 */
class TextRectParser() : PDFTextStripper() {
    val chars = mutableListOf<CharData>()

    init {
        super.setSortByPosition(true)
    }

    /**
     * Clear the chars list, populate it with the characters on the specified
     * page, and returns the list.
     */
    fun getCharsOnPage(doc: PDDocument, page: Int): List<CharData> {
        chars.clear()
        this.startPage = page
        this.endPage = page + 1
        val dummy = OutputStreamWriter(ByteArrayOutputStream())
        writeText(doc, dummy)

        return chars
    }

    @Throws(IOException::class)
    override fun writeString(string: String, textPositions: List<TextPosition>?) {
        for (text in textPositions!!) {
            chars.add(CharData(text.xDirAdj, text.yDirAdj,
                    text.widthDirAdj, text.heightDir))
        }
    }
}


fun main(args: Array<String>) {
    val f = File(args[0])
    val doc = PDDocument.load(f)

    val parser = TextRectParser()
    for (pagenum in 0..doc.numberOfPages - 1) {
        val chars = parser.getCharsOnPage(doc, pagenum)
        val page = doc.getPage(pagenum)
        chars.map {char -> drawRect(doc, page, char)}
    }

    doc.save("modified.pdf")
    doc.close()
}


// Draw a char's bounding box on the specified page
fun drawRect(document: PDDocument, page: PDPage, char: CharData) {
    val pageHeight = page.trimBox.height
    val leftOffset = page.trimBox.lowerLeftX
    val topOffset = page.trimBox.lowerLeftY
    val content = PDPageContentStream(document, page, PDPageContentStream.AppendMode.APPEND, false)
    content.addRect(char.left + leftOffset,
            (pageHeight + topOffset) - char.top,
            char.width, char.height)
    content.setStrokingColor(Color.RED)
    content.stroke()
    content.close()
}


