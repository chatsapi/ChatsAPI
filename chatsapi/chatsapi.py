from typing import Dict, List

import hnswlib
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.llms.llama_api import LlamaAPI
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import spacy
import uuid


def safe_cast(value, target_type, default):
    """
    Safely casts a value to the given target_type. If the conversion fails,
    returns the default value.

    :param value: The value to cast.
    :param target_type: The type to which the value should be cast.
    :param default: The default value to return if conversion fails.
    :return: The cast value or the default value.
    """
    try:
        if target_type == int:
            # Strip non-numeric characters and convert to int
            sanitized_value = ''.join(filter(str.isdigit, str(value)))
            return int(sanitized_value) if sanitized_value else default
        elif target_type == float:
            # Handle float conversion
            sanitized_value = ''.join(filter(lambda c: c.isdigit() or c == '.', str(value)))
            return float(sanitized_value) if sanitized_value else default
        elif target_type == str:
            # Ensure value is a string
            return str(value)
        elif target_type == bool:
            # Convert truthy/falsy values to boolean
            return str(value).strip().lower() in ['true', '1', 'yes', 'y']
        else:
            # Attempt generic casting for other types
            return target_type(value)
    except (ValueError, TypeError):
        # Fallback to the default value
        return default


class ChatsAPI:
    """
    ChatsAPI class to handle chat queries using SBERT embeddings and HNSWlib or BM25 search.
    """

    def __init__(self, llm_type=None, llm_model=None, llm_api_key=None):
        self.llm_agent = None
        self.llm_type = llm_type
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.tokenized_routes = None
        self.routes = []  # Each route will now include metadata
        self.route_functions = {}
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.hnsw_index = None
        self.bm25 = None
        self.initialized = False
        self.nlp = spacy.load("en_core_web_sm")
        self.conversations: Dict[str, List[ChatMessage]] = {}

    # Register a new route
    def trigger(self, route: str):
        """
        Decorator to register a new route.
        """

        def decorator(func):
            # Register the route
            route_info = {"route": route, "function": func, "extract_params": []}

            # If the function has deferred extraction metadata, add it
            if hasattr(func, "_extract_metadata"):
                route_info["extract_params"].append(func._extract_metadata)

            self.routes.append(route_info)
            self.route_functions[route] = func
            return func

        return decorator

    # Add the extract decorator
    def extract(self, params: list):
        """
        Decorator to add multiple parameter extraction metadata to a route.
        :param params: A list of tuples, where each tuple is (key, data_type, default_value)
        """

        def decorator(func):
            # Wrapper to ensure the function is associated with extraction metadata
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            # Add extraction metadata AFTER route registration
            def add_extraction_metadata():
                for route_info in self.routes:
                    if route_info["function"] == func:
                        # Append each parameter as a dictionary
                        for param in params:
                            # Ensure param is a tuple of (key, data_type, default_value)
                            if isinstance(param, tuple) and len(param) == 3:
                                key, data_type, default_value = param
                                route_info["extract_params"].append({
                                    "key": key,
                                    "data_type": data_type,
                                    "default": default_value,
                                })
                            else:
                                raise ValueError(
                                    f"Each parameter must be a tuple of (key, data_type, default_value). Got: {param}"
                                )
                        break

            # Delay the metadata addition until after initialization
            if self.initialized:
                add_extraction_metadata()
            else:
                # Defer until the route system initializes
                wrapper._extract_metadata = params

            return wrapper

        return decorator

    # Initialize the system
    def initialize(self):
        if not self.routes:
            raise ValueError("No routes have been registered.")

        # Generate embeddings for all items
        route_texts = [route_info["route"] for route_info in self.routes]
        self.embeddings = self.model.encode(route_texts, normalize_embeddings=True)

        # Initialize HNSWlib index
        dim = self.embeddings.shape[1]
        self.hnsw_index = hnswlib.Index(space='cosine', dim=dim)
        self.hnsw_index.init_index(max_elements=len(self.embeddings), ef_construction=100, M=16)
        self.hnsw_index.add_items(self.embeddings)

        # Initialize BM25
        self.tokenized_routes = [route.split() for route in route_texts]
        self.bm25 = BM25Okapi(self.tokenized_routes)
        self.initialized = True

    async def run(self, input_text: str, method="bm25_hybrid"):
        """
        Handles chat queries using the specified method: 'hnswlib' or 'bm25_hybrid'.

        :param input_text: User query
        :param method: Search method ('hnswlib' or 'bm25_hybrid')
        :return: The most similar route or a default response
        """

        # Initialize the system if not already done
        if not self.initialized:
            self.initialize()

        # Choose the search method
        if method == "hnswlib":
            return await self.sbert_hnswlib(input_text, self.routes)
        elif method == "bm25_hybrid":
            return await self.sbert_bm25_hybrid(input_text, self.routes)
        else:
            raise ValueError("Invalid method. Choose 'hnswlib' or 'bm25_hybrid'.")

    async def extract_value_for_key(self, input_text, key):
        doc = self.nlp(input_text)
        best_match = None
        max_similarity = 0

        for ent in doc.ents:
            similarity = ent.similarity(self.nlp(key))
            if similarity > max_similarity:
                best_match = ent.text
                max_similarity = similarity

        return best_match

    async def execute_route(self, route_info, input_text):
        """
        Execute the matched route's function with extracted parameters.
        """
        # Initialize the extracted parameters dictionary
        extracted_params = {}

        # Ensure extract_params is a list of parameter groups
        if not isinstance(route_info["extract_params"], list):
            raise ValueError(f"Expected 'extract_params' to be a list, got {type(route_info['extract_params'])}")

        # Iterate through the list of parameter groups
        for param_group in route_info["extract_params"]:
            if not isinstance(param_group, list):
                raise ValueError(f"Expected param_group to be a list of tuples, got {type(param_group)}")

            for param in param_group:
                # Ensure each param is a tuple with three elements
                if not (isinstance(param, tuple) and len(param) == 4):
                    raise ValueError(
                        f"Expected param to be a tuple of (name, key, data_type, default), got {type(param)}")

                name, key, data_type, default = param

                # Check if the entity is extracted
                extracted_value = await self.extract_value_for_key(input_text, key)

                # Use the extracted value if found; otherwise, use the default
                final_value = extracted_value if extracted_value is not None else default

                # Convert the value to the required data type
                try:
                    final_value = safe_cast(final_value, data_type, default)
                except ValueError:
                    raise ValueError(f"Could not convert {final_value} to {data_type}")

                # Store the result
                extracted_params[name] = final_value

        # Call the route's function with the input_text and extracted parameters
        return await route_info["function"](input_text, extracted_params)

    async def sbert_hnswlib(self, input_text: str, choices):
        """
        SBERT + HNSWlib method for fast similarity search.
        """
        # Encode the new input
        new_embedding = self.model.encode(input_text, normalize_embeddings=True)

        # Query HNSWlib index
        top_k = 1
        labels, distances = self.hnsw_index.knn_query(new_embedding, k=top_k)

        # Get the most similar result
        most_similar_idx = labels[0][0]
        similarity_score = 1 - distances[0][0]  # Convert distance to similarity (cosine)

        # Define a similarity threshold
        similarity_threshold = 0.5  # Adjust as needed
        if similarity_score >= similarity_threshold:
            route_info = choices[most_similar_idx]
            return await self.execute_route(route_info, input_text)
        else:
            result = None  # Default output

        # Output the result
        print("Input:", input_text)
        print("Similarity Score:", similarity_score)
        print("Result:", result)
        return result

    async def sbert_bm25_hybrid(self, input_text: str, choices):
        """
        SBERT + BM25 Hybrid method for similarity search.
        """
        # Step 1: Use BM25 to prefilter the top candidates
        tokenized_query = input_text.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k BM25 candidates
        top_k_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:3]
        top_routes = [choices[i] for i in top_k_idx]
        top_embeddings = [self.embeddings[i] for i in top_k_idx]

        # Step 2: Use SBERT to refine BM25 candidates
        new_embedding = self.model.encode(input_text, normalize_embeddings=True)
        cosine_scores = util.cos_sim(new_embedding, top_embeddings)

        # Find the most similar text among BM25-filtered results
        max_score = cosine_scores.max().item()
        best_match_idx = cosine_scores.argmax().item()

        # Define a similarity threshold
        similarity_threshold = 0.5  # Adjust as needed
        if max_score >= similarity_threshold:

            # print("BM25 Top Candidates:", top_routes)
            print("Max Cosine Similarity Score:", max_score)

            route_info = top_routes[best_match_idx]
            return await self.execute_route(route_info, input_text)
        else:
            result = None  # Default output

        # Output the result
        # print("BM25 Top Candidates:", top_routes)
        print("Max Cosine Similarity Score:", max_score)
        return result

    """
    Conversation AI
    """

    def _initialize_llm(self, llm_type):
        """
        Initialize the Language Model based on the specified type.
        """
        llm_classes = {
            "gemini": Gemini,
            "openai": OpenAI,
            "llama_api": LlamaAPI,
            "ollama": Ollama,
        }
        if llm_type in llm_classes:
            return llm_classes[llm_type](model=self.llm_model, api_key=self.llm_api_key)
        raise ValueError(f"Unsupported LLM type: {llm_type}")

    async def query(self, input_text: str):
        if self.llm_agent is None:
            self.llm_agent = self._initialize_llm(self.llm_type)

        user_message = ChatMessage(
            role="user",
            content=(
                f"""Message: {input_text}"""
            )
        )

        local_run = await self.run(input_text)

        if local_run is not None:
            return local_run

        # Generate a response from Gemini
        resp = self.llm_agent.chat([user_message])

        return {
            "role": resp.message.role,
            "message": resp.message.content,
        }

    async def conversation(self, input_text: str, session_id: str):
        if not session_id:
            return {"error": "Session ID not found. Please start a new session using /set-session"}

        # Initialize session conversation if not exists
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        user_message = ChatMessage(
            role="user",
            content=(
                f"""Message: {input_text}"""
            )
        )

        self.conversations[session_id].append(user_message)

        local_run = await self.run(input_text)

        if local_run is not None:
            return local_run

        # Generate a response from Gemini
        resp = self.llm_agent.chat(self.conversations[session_id])

        # Add AI's response to the conversation
        ai_message = ChatMessage(role=resp.message.role, content=resp.message.content)
        self.conversations[session_id].append(ai_message)

        return {
            "role": resp.message.role,
            "message": resp.message.content,
        }

    def set_session(self):
        self.llm_agent = self._initialize_llm(self.llm_type)

        session_id = str(uuid.uuid4())  # Generate a unique session ID
        self.conversations[session_id] = []  # Initialize conversation for the session

        # Add a system message to instruct the LLM to give short responses
        instruction_message = ChatMessage(
            role="system",
            content="Please keep your responses short and concise and to the point."
        )
        self.conversations[session_id].append(instruction_message)
        return session_id

    def end_session(self, session_id):
        if session_id in self.conversations:
            del self.conversations[session_id]
            return {"message": "Session reset"}
        return {"message": "Session not found"}
