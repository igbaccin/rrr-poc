import unittest

import rrr.writer as writer


HOSTED = hasattr(writer, "_WRITER_HOST_EXECUTION_PROFILE")


@unittest.skipUnless(HOSTED, "needs the hosted writer")
class HostedTreeTests(unittest.TestCase):
    def test_the_hosted_writer_is_the_one_under_test(self):
        self.assertTrue(hasattr(writer, "_audit_raw_citation_contract"))
        self.assertFalse(hasattr(writer, "_normalize_leaked_doc_id_citations"))

    def test_the_hosted_modules_import_together(self):
        import rrr.reasoner
        import rrr.render as render

        self.assertTrue(callable(render.parse_citations))


if __name__ == "__main__":
    unittest.main()
