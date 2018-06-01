# Comments MM June 1 on Thesis Maarten de Jonge

1. 2elezer = van someren: zet op voorblad
2. de titel is bijna goed, maar zins classificatie is misleidend, want dat doe je niet echt. Dus verander
3. abstract is heel goed en lekker puntig.
4. l44 _classify sentences_ dit doe je toch niet echt? Jij doet echt "regels", _lines of text_
5. **figuur 1** Dat kan wel wat dieper: root= meeting en onder speech zou je nog "paragraaf" kunne zetten
6. **Introductie** heel goed geschreven en je legt het duidelijk uit. Probeer jouw probleem wel in de literatuur te plaatsen, ook hier al. In je literature review kan je er dieper op in gaan. Ik zie minstens 2 richtingen die je moet behandelen:
	7. structuur en layout herkenning op tekst (bijv wat is precies de tekst van 1 krantenartikel op een woest opgemaakte pagina)?
	8. Scraping en text extractie op webpaginas (bijv die wrapper inductie, die ook probeert te generaliseren)
	9. Ook zou je hier al over supervised en unsupervised aanpakken kunen spreken en laten zien wat het in deze setting betekent.
	
## sectie 2

1. l77 	volgens mij bestaat de bondsdag sinds 49, en heette het daarvoor reichstag. De reichtstag  proceedings werden ook al gepubliceerd. Dus maak dit precies.
2. Die Figuur 2 en 3 zijn erg goed om je probeelm duidelijk te maken
3. l90 zin loopy niet lekker en lijkt ook ambigu
4. Sectie 2.1 is een brei van informatie die allemaal best leuk is maar niet goed gestructureerd. Zorg eerst eens dat je weet wat je wilt vertellen, en deel dat op in blokjes die bij elkaar horen, en ga dan pas zinnen schrijven. Dit is nu niet goed. 
	5. Het is best lastig. 
	6. Je hebt een bron dataset, je pdf.s. Begin daar eerst wat over te vertellen. Hoeveel, hoe lang, hoeveel woorden, ruwweg hoeveel speeches, etc.
	7. Dan "converteer" je die op 2 manieren naar "text plus coordinaten". Leg uit wat je verliest (lettertypes, groottes, etc, etc), 
		* op het niveau van characters
		* op het niveau van tekstregels (waarbij kolommen in acht worden genomen (zo goed als dat lukt) )
	8. Dan ga je 1 daarvan nog op 2 manieren annoteren:
		* met je regelgebaseerde systeem
		* en een setje daarvan nog een keer met de hand.
	9. Ben ik er zo uit? Ik wete zeker dat een andere lezer dit er echt niet uitkrijgt. Doe je best ;-)
* Figgur 4: je hebt encosding probleem in je XML. Gebruik je die `<b>` tags? Anders zou ik het eruit laten.
* l140 kan er wat mij betrfet uit. Ik hou niet van "motivatue/excuus" achtige zinnen. Je bullet points geven zelf aan dat dit een goede vraag is. 
* Het is vreemd om een onderzoeksvraag niet met een punt te eindigen. 
* Ook kan je een sectie niet enidigenmet een bulleted list. Dus hier moet nog wat komen.

## Setie 3

1. Nice die referentie [5]. Je kan nu ook al het verband met jouw werk leggen. In hoeveer doe jij iets anders? Het is in ieder geval dezelfde lexicale+spatiale combinatie. Leuk! Ga hier gewoon wat dieper op in.
2. Je hebt het over text en zins classificatie, maar het maakt nogal uit naar wat voor type klasses je classificeert. Benoem dat en ga daar dieper op in. In jouw geval heb je het over structurele documenten. Ik zou zeggen dat dit eerder lijkt op een sentence tokenizer of een paragraph detector dan op klassiek tekst classificatie (eg spam detectie). 
3. 179-198 Dit blijft allemaal veel te abstract en higjh-level om ook maar iets te zeggen. Bijv _where the input words are tokenized and embedded_. Ten eerste wordt de tekst getokenizeerd en niet de woorden. Maar is het duidelijk wat het betekend dat woorden _embedded_ wprden? Het is beter om te schrijven wat de bedoeling is achter die embedding. Iets als _words are mapped to  real valued vectors in a low dimensional space which are supposed to capture aspects f their meaning (eg, words with similar meaning have (cosine) similar vectors)_ 
4.

#
# Bibs

* Dit is niet goed en netjes. Refereer naar de journal of conferentie versies en niet naar preprints. Bijvoorbeeld: <https://www.sciencedirect.com/science/article/pii/S0950705114002640> 
* Zorg dat het er perfect utziet.
