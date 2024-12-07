import os
import unittest
from chatsapi import ChatsAPI
from dotenv import load_dotenv

load_dotenv()


class TestChatsAPI(unittest.TestCase):
    def setUp(self):
        """
        Set up a new instance of ChatsAPI for testing.
        """
        self.api = ChatsAPI(
            llm_type="gemini",
            llm_model="models/gemini-pro",
            llm_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    def test_initialization(self):
        """
        Test that the ChatsAPI instance initializes correctly.
        """
        self.assertIsInstance(self.api, ChatsAPI)
        self.assertIsNotNone(self.api.routes)
        self.assertIsNotNone(self.api.model)

    def test_register_route(self):
        """
        Test registering a route using the `trigger` decorator.
        """

        @self.api.trigger("/greet")
        def greet_route(input_text, params):
            return {"response": "Hello, world!"}

        self.assertIn("/greet", [route["route"] for route in self.api.routes])

    def test_extract_decorator(self):
        """
        Test the `extract` decorator with parameter extraction.
        """

        @self.api.trigger("/extract")
        @self.api.extract([("name", str, "default_name"), ("age", int, 25)])
        def extract_route(input_text, params):
            return {"name": params["name"], "age": params["age"]}

        self.assertIn("/extract", [route["route"] for route in self.api.routes])
        route_info = next((r for r in self.api.routes if r["route"] == "/extract"), None)
        self.assertIsNotNone(route_info)
        self.assertEqual(len(route_info["extract_params"]), 1)

    def test_sbert_hnswlib(self):
        """
        Test the SBERT + HNSWlib similarity search.
        """
        @self.api.trigger("/hello")
        def hello_route(input_text, params):
            return {"response": "Hello, user!"}

        self.api.initialize()
        input_text = "Hi there!"
        result = self.api.run(input_text, method="hnswlib")
        self.assertIsNotNone(result)

    def test_sbert_bm25_hybrid(self):
        """
        Test the SBERT + BM25 hybrid similarity search.
        """
        @self.api.trigger("/test")
        def test_route(input_text, params):
            return {"response": "Testing route."}

        self.api.initialize()
        input_text = "Run test route."
        result = self.api.run(input_text, method="bm25_hybrid")
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
