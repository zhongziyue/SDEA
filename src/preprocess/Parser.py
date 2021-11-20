import re
from enum import Enum
from typing import List, Tuple

from tools.MultiprocessingTool import MultiprocessingTool

pref = {
    "rdf:": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs:": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd:": "http://www.w3.org/2001/XMLSchema#",
    "owl:": "http://www.w3.org/2002/07/owl#", "skos:": "http://www.w3.org/2004/02/skos/core#",
    "dc:": "http://purl.org/dc/terms/",
    "foaf:": "http://xmlns.com/foaf/0.1/",
    "vcard:": "http://www.w3.org/2006/vcard/ns#",
    "dbp:": "http://dbpedia.org/",
    "y1:": "http://www.mpii.de/yago/resource/",
    "y2:": "http://yago-knowledge.org/resource/",
    "geo:": "http://www.geonames.org/ontology#",
    'wiki:': 'http://www.wikidata.org/',
    'schema:': 'http://schema.org/',
    'freebase:': 'http://rdf.freebase.com/',
    'dbp_zh': 'http://zh.dbpedia.org/',
    'dbp_fr': 'http://fr.dbpedia.org/',
    'dbp_ja': 'http://ja.dbpedia.org/',
}


class OEAFileType(Enum):
    attr = 0
    rel = 1
    ttl_full = 2


def strip_square_brackets(s):
    # s = ""
    if s.startswith('"'):
        rindex = s.rfind('"')
        if rindex > 0:
            s = s[:rindex + 1]
    else:
        if s.startswith('<'):
            s = s[1:]
        if s.endswith('>'):
            s = s[:-1]
    return s


def compress_uri(uri):
    uri = strip_square_brackets(uri)
    if uri.startswith("http://"):
        for key, val in pref.items():
            if uri.startswith(val):
                uri = uri.replace(val, key)
    return uri


def oea_attr_line(line: str):
    fact: List[str] = line.strip('\n').split('\t')
    if not fact[2].startswith('"'):
        fact[2] = ''.join(('"', fact[2], '"'))
    return compress_uri(fact[0]), compress_uri(fact[1]), compress_uri(fact[2])


def oea_rel_line(line: str) -> Tuple:
    fact: List[str] = line.strip('\n').split('\t')
    return compress_uri(fact[0]), compress_uri(fact[1]), compress_uri(fact[2])


def oea_truth_line(line: str) -> Tuple:
    fact: List[str] = line.strip().split('\t')
    return compress_uri(fact[0]), compress_uri(fact[1])


ttlPattern = "([^\\s]+)\\s+([^\\s]+)\\s+(.+)\\s*."


def stripSquareBrackets(s):
    # s = ""
    if s.startswith('"'):
        rindex = s.rfind('"')
        if rindex > 0:
            s = s[:rindex + 1]
    else:
        if s.startswith('<'):
            s = s[1:]
        if s.endswith('>'):
            s = s[:-1]
    return s


def ttl_no_compress_line(line):
    if line.startswith('#'):
        return None, None, None
    fact = re.match(ttlPattern, line.rstrip())
    if fact is None:
        print(line)
    sbj = stripSquareBrackets(fact[1])
    pred = stripSquareBrackets(fact[2])
    obj = stripSquareBrackets(fact[3])
    return sbj, pred, obj


def for_file(file, file_type: OEAFileType) -> list:
    line_solver = None
    if file_type == OEAFileType.attr:
        line_solver = oea_attr_line
    elif file_type == OEAFileType.rel:
        line_solver = oea_rel_line
    elif file_type == OEAFileType.ttl_full:
        line_solver = ttl_no_compress_line
    assert line_solver is not None
    with open(file, 'r', encoding='utf-8') as rfile:
        mt = MultiprocessingTool()
        results = mt.packed_solver(line_solver).send_packs(rfile).receive_results()
        results = [triple for triple in results if triple[0] is not None]
    return results
