import tempfile
import unittest
from pathlib import Path

import rrr.render as render
from rrr.ingest import (
    _ascii_fold,
    _bib_surnames_list,
    _bibtex_plain,
    _clean_surname_token,
    generate_doc_id,
    parse_bib_sidecar,
)


BIB = r"""
@techreport{WorldBank1995,
  author = {{World Bank}},
  title = {{G}hana: Poverty Past, Present and Future},
  institution = {World Bank},
  year = {1995}
}
@techreport{GSS2000,
  author = {{Ghana Statistical Service}},
  title = {Poverty Trends in {G}hana in the 1990s},
  year = {2000}
}
@article{deHaas2017,
  author = {de Haas, Michiel},
  title = {Measuring rural welfare in colonial Africa},
  journal = {The Economic History Review},
  year = {2017}
}
@article{vanderGeest2011,
  author = {van der Geest, Kees},
  title = {North-South Migration in Ghana: What Role for the Environment?},
  journal = {International Migration},
  year = {2011}
}
@article{FrankemaWaijenburg2012,
  author = {Frankema, Ewout and van Waijenburg, Marlous},
  title = {Structural Impediments to African Growth?},
  journal = {The Journal of Economic History},
  year = {2012}
}
@article{MeierWeisdorf2016,
  author = {Meier zu Selhausen, Felix and Weisdorf, Jacob},
  title = {A colonial legacy of African gender inequality?},
  journal = {The Economic History Review},
  year = {2016}
}
@techreport{CanagarajahPortner2003,
  author = {Canagarajah, Sudharshan and P{\"o}rtner, Claus C.},
  title = {Evolution of Poverty and Welfare in Ghana in the 1990s},
  year = {2003}
}
@article{Ronnback2014,
  author = {Rönnbäck, Klas},
  title = {Living standards on the pre-colonial Gold Coast},
  journal = {European Review of Economic History},
  year = {2014}
}
@article{AryeeteyAttoh1988,
  author = {Aryeetey-Attoh, Samuel and Chatterjee, Lata},
  title = {Regional inequalities in Ghana: Assessment and policy issues},
  journal = {Tijdschrift voor Economische en Sociale Geografie},
  year = {1988}
}
@book{AppiahKubi2008,
  author = {Appiah-Kubi, Kojo and Oduro, Abena and Senadza, Bernardin},
  title = {Understanding poverty in {G}hana: risk and vulnerability},
  year = {2008}
}
"""


class BibAuthorNameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bibliography.bib"
            path.write_text(BIB, encoding="utf-8")
            cls.entries = {entry["key"]: entry for entry in parse_bib_sidecar(path)}

    def _doc_id(self, key):
        entry = self.entries[key]
        return generate_doc_id(_bib_surnames_list(entry), entry["year"])

    def test_protective_braces_do_not_reach_the_catalogue(self):
        world_bank = self.entries["WorldBank1995"]
        self.assertEqual("World Bank", world_bank["authors_short"])
        self.assertEqual("World Bank", world_bank["first_surname"])
        self.assertEqual("World Bank", world_bank["author_full"])
        self.assertEqual("Ghana: Poverty Past, Present and Future", world_bank["title"])
        service = self.entries["GSS2000"]
        self.assertEqual("Ghana Statistical Service", service["first_surname"])
        self.assertEqual("Poverty Trends in Ghana in the 1990s", service["title"])
        for entry in self.entries.values():
            for field in ("authors_short", "author_full", "first_surname", "title"):
                self.assertNotIn("{", entry[field], (entry["key"], field))
                self.assertNotIn("}", entry[field], (entry["key"], field))
        self.assertEqual("WorldBank_1995", self._doc_id("WorldBank1995"))
        self.assertEqual("GhanaStatisticalService_2000", self._doc_id("GSS2000"))

    def test_the_particle_stays_with_the_surname(self):
        self.assertEqual("de Haas", self.entries["deHaas2017"]["first_surname"])
        self.assertEqual("de Haas, Michiel", self.entries["deHaas2017"]["author_full"])
        self.assertEqual("van der Geest", self.entries["vanderGeest2011"]["first_surname"])
        self.assertEqual(
            "Frankema & van Waijenburg",
            self.entries["FrankemaWaijenburg2012"]["authors_short"],
        )
        self.assertEqual(
            ["Meier zu Selhausen", "Weisdorf"],
            self.entries["MeierWeisdorf2016"]["surnames"],
        )
        self.assertEqual("deHaas_2017", self._doc_id("deHaas2017"))
        self.assertEqual("vanderGeest_2011", self._doc_id("vanderGeest2011"))
        self.assertEqual(
            "Frankema&vanWaijenburg_2012", self._doc_id("FrankemaWaijenburg2012")
        )
        self.assertEqual(
            "MeierzuSelhausen&Weisdorf_2016", self._doc_id("MeierWeisdorf2016")
        )

    def test_accented_letters_are_transliterated_in_the_doc_id(self):
        portner = self.entries["CanagarajahPortner2003"]
        self.assertEqual("Canagarajah & Pörtner", portner["authors_short"])
        self.assertEqual("Canagarajah&Portner_2003", self._doc_id("CanagarajahPortner2003"))
        self.assertEqual("Rönnbäck", self.entries["Ronnback2014"]["first_surname"])
        self.assertEqual("Ronnback_2014", self._doc_id("Ronnback2014"))

    def test_hyphenated_surnames_stay_whole(self):
        entry = self.entries["AryeeteyAttoh1988"]
        self.assertEqual("Aryeetey-Attoh & Chatterjee", entry["authors_short"])
        self.assertEqual("Aryeetey-Attoh", entry["first_surname"])
        self.assertEqual("AryeeteyAttoh&Chatterjee_1988", self._doc_id("AryeeteyAttoh1988"))
        self.assertEqual("AppiahKubiEtAl_2008", self._doc_id("AppiahKubi2008"))

    def test_every_doc_id_gives_a_label_the_parser_reads_back(self):
        doc_ids = [self._doc_id(key) for key in self.entries]
        self.assertEqual(len(doc_ids), len(set(doc_ids)))
        render.set_metadata_labels(
            [{"doc_id": doc_id, "year": doc_id.rsplit("_", 1)[1]} for doc_id in doc_ids]
        )
        try:
            lookup = render._build_display_lookup(doc_ids)
            for doc_id in doc_ids:
                narrative = render.render_citation(doc_id, 7)
                label, _, tail = narrative.rpartition(" (")
                for text in (f"As {narrative} shows.", f"It holds ({label} {tail}."):
                    found = [
                        c["doc_id"]
                        for c in render.parse_citations(text, display_lookup=lookup)
                    ]
                    self.assertEqual([doc_id], found, text)
        finally:
            render.set_metadata_labels([])


class NameHelperTests(unittest.TestCase):
    def test_ascii_fold(self):
        self.assertEqual("Portner", _ascii_fold("Pörtner"))
        self.assertEqual("Ronnback", _ascii_fold("Rönnbäck"))
        self.assertEqual("Orsted", _ascii_fold("Ørsted"))
        self.assertEqual("Strasse", _ascii_fold("Straße"))
        self.assertEqual("Wojcik", _ascii_fold("Wójcik"))
        self.assertEqual("Lukasz", _ascii_fold("Łukasz"))
        self.assertEqual("efficient", _ascii_fold("eﬃcient"))

    def test_clean_surname_token(self):
        self.assertEqual("Portner", _clean_surname_token("Pörtner"))
        self.assertEqual("vanZanden", _clean_surname_token("Van Zanden"))
        self.assertEqual("deHaas", _clean_surname_token("De Haas"))
        self.assertEqual("vanderGeest", _clean_surname_token("van der Geest"))
        self.assertEqual("AryeeteyAttoh", _clean_surname_token("Aryeetey-Attoh"))
        self.assertEqual("ORourke", _clean_surname_token("O'Rourke"))

    def test_bibtex_plain(self):
        self.assertEqual("Pörtner", _bibtex_plain(r'P{\"o}rtner'))
        self.assertEqual("Pörtner", _bibtex_plain(r'P\"ortner'))
        self.assertEqual("Rönnbäck", _bibtex_plain(r'R{\"o}nnb{\"a}ck'))
        self.assertEqual("World Bank", _bibtex_plain("{World Bank}"))
        self.assertEqual("François", _bibtex_plain(r"Fran\c{c}ois"))
        self.assertEqual("Research & Development", _bibtex_plain(r"Research \& Development"))
        self.assertEqual("Ghana's North", _bibtex_plain("{G}hana's {N}orth"))
        self.assertEqual("", _bibtex_plain(None))

    def test_doc_id_of_a_name_with_no_latin_letters_is_empty(self):
        self.assertEqual("", generate_doc_id(["王"], "2001"))


if __name__ == "__main__":
    unittest.main()
