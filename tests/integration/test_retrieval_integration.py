import unittest
from src.retrieval.vector_store import get_vector_store
from src.retrieval.retriever import retrieve_documents

class TestRetrievalIntegration(unittest.TestCase):
    def test_chroma_connection(self):
        store = get_vector_store()
        self.assertIsNotNone(store)

    def test_retrieval_leaves(self):
        res = retrieve_documents("How many annual leave days are provided?", k=2)
        self.assertGreater(len(res.chunks), 0)
        self.assertIn("leave_policy.txt", res.sources)
        self.assertIn("20 days", res.chunks[0].content)

    def test_empty_query_fallback(self):
        res = retrieve_documents("xyzrandomnonexistentquery12345", k=1)
        self.assertIsNotNone(res)

if __name__ == "__main__":
    unittest.main()
