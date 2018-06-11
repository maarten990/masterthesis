import re
import sys

from lxml import etree
import urwid
import pandas as pd


def handle_input(key, data=None):
    if data is not None:
        node, speaker = data
        parent = node.getparent()
        parent.attrib["is-speech"] = "true"
        parent.attrib["speaker"] = speaker
    raise urwid.ExitMainLoop()


parser = etree.XMLParser(ns_clean=True, encoding="utf-8")
with open(sys.argv[1], "r", encoding="utf-8") as f:
    xml = etree.fromstring(f.read().encode("utf-8"), parser)

wiki_url = (
    "https://de.wikipedia.org/wiki/Liste_der_Mitglieder_des_Deutschen_Bundestages_({:02d}._Wahlperiode)"
)

# the set of extracted speakers to be compared to all present speakers
speakers = set(
    [
        re.sub(r"Dr\.", "", speaker).strip()
        for speaker in xml.xpath('/pdf2xml/page/text[@is-speech="true"]/@speaker')
    ]
)
start_of_proceedings = xml.xpath(
    r'/pdf2xml/page/text[re:test(text(), "^\s*Beginn:?\s*[\d\. ]+\s*Uhr")]',
    namespaces={"re": "http://exslt.org/regular-expressions"},
)[0]

toc = [
    " ".join(node.xpath(".//text()"))
    for node in start_of_proceedings.xpath(".//preceding::text")
]

tables = pd.io.html.read_html(
    wiki_url.format(int(sys.argv[2])), attrs={"class": "wikitable"}, parse_dates=False
)

members = tables[-1][0].tolist()
toc_text = " ".join(toc)
expected_speakers = set(
    [
        re.sub(r"\([^)]+\)", "", member).strip()
        for member in members
        if member in toc_text
    ]
)

for spkr in speakers:
    for exp in expected_speakers:
        if spkr in exp:
            expected_speakers.remove(exp)
            break

for speaker in expected_speakers:
    for node in start_of_proceedings.xpath(
        f'./following::b[contains(text(), "{speaker}")]'
    ):
        disp_node = node.getparent()
        if "is-speech" in disp_node.attrib and disp_node.attrib["is-speech"] == "true":
            continue

        for _ in range(3):
            prev = disp_node.getprevious()
            if prev is not None:
                disp_node = prev
            else:
                break

        lines = [
            urwid.Text(f"Potential speaker: {speaker}", align="left"),
            urwid.Text(""),
            urwid.Text(""),
            urwid.Text(""),
        ]
        for i in range(7):
            contents = "".join(disp_node.xpath(".//text()"))
            if i == 3:
                txt = urwid.Text(("bold", contents), align="left")
            else:
                txt = urwid.Text(contents, align="left")
            lines.append(txt)
            next = disp_node.getnext()
            if next is not None:
                disp_node = next
            else:
                break

        lines.append(urwid.Text(""))
        lines.append(
            urwid.Columns(
                [
                    urwid.Button("Not a speech", handle_input),
                    urwid.Button("Speech", handle_input, (node, speaker)),
                ]
            )
        )
        fill = urwid.Filler(urwid.Pile(lines), "top")
        loop = urwid.MainLoop(fill)
        loop.run()

with open("out.xml", "w") as f:
    f.write(etree.tostring(xml, pretty_print=True, encoding="utf-8").decode())
