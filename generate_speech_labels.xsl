<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:xs="http://www.w3.org/2001/XMLSchema"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:pm="http://www.politicalmashup.nl"
    xmlns:dcterms="http://purl.org/dc/terms/"
    exclude-result-prefixes="xs" version="2.0">

  
    <xsl:strip-space elements="*"/>
    <xsl:output method="xml" indent="yes"/>

    <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:apply-templates select="@* | node()"/>
        </xsl:copy>
    </xsl:template>

    <xsl:template match="text">
        <xsl:variable name="speech" select=".[b] and pm:isa-speech-or-stage-direction(./b[1])"/>
        <xsl:copy>
            <xsl:attribute name="is-speech">
                <xsl:value-of select="$speech"/>
            </xsl:attribute>
            <xsl:if test="$speech">
                <xsl:attribute name="speaker">
                    <xsl:value-of select="pm:speaker-in-speech(.)"/>
                </xsl:attribute>
            </xsl:if>
            <xsl:apply-templates select="@* | node()"/>
        </xsl:copy>
    </xsl:template>

    <xsl:variable name="start-of-index" select="/pdf2xml/page/text[matches(string-join((., following::text[1], following::text[2]), ''),
                                                               '^\s*Beginn:?\s*[\d\. ]+\s*Uhr')][1]"/>
    <xsl:variable name="index" select="$start-of-index/preceding::text"/>
    <xsl:variable name="body" select="/pdf2xml/page/text[matches(string(.), 'Beginn:? [\d\. ]+ Uhr')][1]/following::*
                                      intersect
                                      /pdf2xml/page/text[matches(., '^\(Schluss:? .+ Uhr\)\s*$')][last()]/preceding::*"/>

    <xsl:function name="pm:isa-speech-or-stage-direction" as="xs:boolean">
        <xsl:param name="node" as="element()"/>
        
        <!-- 
            A speech node:
            1) is bold
            2) is inside the main body of the document (after the node marked 'Beginn:')
            3) either ends with a colon (:) or is followed by a line ending with a colon
            4) isn't preceded by another bold node (i.e. isn't part of a piece of bold text)
            5) doesn't contain the word 'Drucksache', nor does the following node
        -->
        
        <xsl:variable name="format-matches" select="name($node) = 'b' and
                             count($body intersect $node) = 1 and
                             pm:matches-in-n($node, ':\s*$', 4) and
                             pm:match-one-of(string-join(pm:collect-until-match($node, ':\s*$'), ' '),
                                             ('(\([^\)]+\))', ',', '^([aA]lters)?([vV]ize)?[pP]räsident(in)?')) and
                             not(name($node/preceding::*[1]) = 'b') and
                             not(matches(string($node), 'Drucksache') or matches(string($node/following::text[1]), 'Drucksache'))"/>
        
        <!-- As an additional check, make sure the speaker parses properly. -->
        <xsl:choose>
            <xsl:when test="$format-matches">
                <xsl:variable name="full-string" select="string-join(pm:collect-until-match($node, ':\s*$'), ' ')"/>
                <xsl:variable name="party" select="pm:party-in-speech($node)"/>
                <xsl:variable name="function" select="pm:function-in-speech($node)"/>
                
                <xsl:variable name="known-party" select="$party = '' or pm:match-one-of(pm:remove-whitespace($party),
                    ('^BÜNDNIS90/DIEGRÜNEN$', '^CDU/CSU$', '^DIELINKE$', '^SPD$', '^GRÜNE$', '^FDP$', '^BMU$'))"/>
                
                <!-- If the party is unknown, still accept it if the length is below 20 but print a warning. -->
                <xsl:variable name="accepted-party" select='$known-party or 20 > string-length(pm:remove-whitespace($party))'/>
                
                <xsl:variable name="known-function" select="$function = 'De Duitser' or pm:match-one-of(lower-case($function),
                    ('minister', 'staatssekretär', 'wehrbeauftragter', 'bundeskanzler', 'senator', 'präsident', 'beauftragte'))"/>
                
                <!-- Either the party or function has to be empty. -->
                <xsl:variable name="mutex" select="(not($party = '') and $function = 'De Duitser') or ($party = '' and not($function = 'De Duitser'))"/>
                
                <xsl:copy-of select="$known-party and $mutex and $known-function"/>
            </xsl:when>
            <xsl:otherwise>
                <xsl:copy-of select="false()"/>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:function>

    <xsl:function name="pm:party-in-speech" as="xs:string">
        <xsl:param name="speech"/>
        <xsl:variable name="whole-string" select="string-join(pm:collect-until-match($speech, ':\s*$'), ' ')"/>

        <xsl:choose>
            <xsl:when test="matches($whole-string, '\([^\(]+\)') and not(contains($whole-string, ','))">
                <!-- The regex consists of 2 seperate groups of parentheses filled with text, the first of which
                     is optional. While most lines will be of the form ' (Partyname) ', we have to account for the rare-ish
                     case of ' (District) (Partyname) ' -->
                <xsl:copy-of select="normalize-space( replace($whole-string,
                    '([^\(]+)?(\([^\(]+\))?\s*\(([^\(]+)\)\s*:', '$3') )"/>
            </xsl:when>
            <xsl:otherwise>
                <xsl:copy-of select="''"/>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:function>
    
    <xsl:function name="pm:function-in-speech" as="xs:string">
        <xsl:param name="speech"/>
        <xsl:variable name="whole-string" select="string-join(pm:collect-until-match($speech, ':\s*$'), ' ')"/>
        
        <xsl:choose>
            <xsl:when test="matches($whole-string, 'Präsident')">
                <xsl:copy-of select="'Präsident'"/>
            </xsl:when>
            <xsl:when test="matches($whole-string, 'Vizepräsident')">
                <xsl:copy-of select="'Vizepräsident'"/>
            </xsl:when>
            <xsl:when test="matches($whole-string, 'Alterspräsident')">
                <xsl:copy-of select="'Alterspräsident'"/>
            </xsl:when>
            <xsl:when test="matches($whole-string, ',')">
                <xsl:variable name="tokens" select="tokenize($whole-string, ',')"/>
                <xsl:variable name="function" select="string-join(subsequence($tokens, 2, count($tokens)), ' ')"/>
                <!-- Remove spaces in front of periods and remove the trailing colon -->
                <xsl:copy-of select="replace(replace(normalize-space($function), ' \.', '.'), ':\s*$', '')"></xsl:copy-of>
            </xsl:when>
            <xsl:otherwise>
                <xsl:copy-of select="'De Duitser'"/>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:function>

    <!-- Return a sequence of following nodes from the given node up to and including the first node
         that matches the regex. -->
    <xsl:function name="pm:collect-until-match" as="element()*">
        <xsl:param name="node" as="element()"/>
        <xsl:param name="regex" as="xs:string"/>
        
        <xsl:choose>
            <xsl:when test="matches(string($node), $regex)">
                <xsl:sequence select="($node)"/>
            </xsl:when>
            <xsl:otherwise>
                <xsl:variable name="first_match" select="$node/following::text[matches(string(.), $regex)][1]"/>
                <xsl:copy-of select="($node, $node/following::text intersect $first_match/preceding::text, $first_match)"/>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:function>
    
    <!-- Returns whether the given regex is matched within n following text-nodes of the given node. -->
    <xsl:function name="pm:matches-in-n" as="xs:boolean">
        <xsl:param name="node" as="element()"/>
        <xsl:param name="regex" as="xs:string"/>
        <xsl:param name="n" as="xs:integer"/>
        
        <xsl:copy-of select="count(($node, $node/following::text intersect $node/following::text[$n]/preceding::text)[matches(string(.), $regex)]) > 0"/>
    </xsl:function>    

    <xsl:function name="pm:match-one-of" as="xs:boolean">
        <xsl:param name="text" as="xs:string"/>
        <xsl:param name="regexes" as="xs:string*"/>
        <xsl:choose>
            <xsl:when test="count($regexes) = 1">
                <xsl:copy-of select="matches($text, $regexes[1])"/>
            </xsl:when>
            <xsl:otherwise>
                <xsl:copy-of select="matches($text, $regexes[1]) or pm:match-one-of($text, subsequence($regexes, 2))"/>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:function>

    <xsl:function name="pm:remove-whitespace" as="xs:string">
        <xsl:param name="s" as="xs:string"/>
        <xsl:copy-of select="replace($s, '\s+', '')"/>
    </xsl:function>

    <xsl:function name="pm:speaker-in-speech" as="xs:string">
        <xsl:param name="speech"/>
        <xsl:variable name="full-text" select="string-join(pm:collect-until-match($speech, ':'), ' ')"/>
        
        <xsl:choose>
            <!-- Case of 'Präsident John Johnson:' -->
            <xsl:when test="matches($full-text, '[Pp]räsident')">
                  <xsl:variable name="elements" select="subsequence(tokenize($full-text, ' '), 2)"/>
                <xsl:variable name="name" select="string-join($elements, ' ')"/>
                
                <!-- filter any trailing punctuation or whitespace before returning the name -->
                <xsl:copy-of select="normalize-space(replace($name, '[,:]+$', ''))"/>
            </xsl:when>
            
            <!-- Case of 'John Johnson, Bundesminister für blah blah' -->
            <xsl:when test="contains($full-text, ',')">
                <xsl:copy-of select="normalize-space(replace(string($full-text), '^([^,]+),.*$', '$1'))"/>
            </xsl:when>
            
            <!-- Case of 'John Johnson (optional placename) (Partyname):' -->
            <xsl:when test="contains($full-text, '(')">
                <xsl:variable name="name" select="replace($full-text, '([^(]*)\(.*', '$1')"/>
                
                <!-- filter any trailing punctuation or whitespace before returning the name -->
                <xsl:copy-of select="normalize-space(replace($name, '[,:]+$', ''))"/>
            </xsl:when>
            
            <!-- This shouldn't happen -->
            <xsl:otherwise>
                <xsl:copy-of select="string-join(('Could not parse name: ', $full-text), '')"/>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:function>
</xsl:stylesheet>