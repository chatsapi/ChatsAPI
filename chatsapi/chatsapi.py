import hnswlib
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import spacy


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

    def __init__(self):
        self.tokenized_routes = None
        self.routes = []  # Each route will now include metadata
        self.route_functions = {}
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.hnsw_index = None
        self.bm25 = None
        self.nlp = spacy.load("en_core_web_sm")  # Load SpaCy NLP model
        self.initialized = False

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
    def extract(self, key: str, data_type: type, default_value):
        """
        Decorator to add parameter extraction metadata to a route.
        """

        def decorator(func):
            # Wrapper to ensure the function is associated with extraction metadata
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            # Add extraction metadata AFTER route registration
            def add_extraction_metadata():
                for route_info in self.routes:
                    if route_info["function"] == func:
                        route_info["extract_params"].append({
                            "key": key,
                            "data_type": data_type,
                            "default": default_value,
                        })
                        break

            # Delay the metadata addition until after initialization
            if self.initialized:
                add_extraction_metadata()
            else:
                # Defer until the route system initializes
                wrapper._extract_metadata = {
                    "key": key,
                    "data_type": data_type,
                    "default": default_value,
                }

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
            return await self.sbert_hnswlib(input_text)
        elif method == "bm25_hybrid":
            return await self.sbert_bm25_hybrid(input_text)
        else:
            raise ValueError("Invalid method. Choose 'hnswlib' or 'bm25_hybrid'.")

    async def execute_route(self, route_info, input_text):
        """
        Execute the matched route's function with extracted parameters.
        """
        # Parse the input_text using SpaCy
        doc = self.nlp(input_text)

        # Initialize the extracted parameters dictionary
        extracted_params = {}

        for param in route_info["extract_params"]:
            key = param["key"]
            data_type = param["data_type"]
            default = param["default"]

            # Attempt to extract the value from input_text
            extracted_value = None

            # Example logic: Match based on parameter name or key
            for ent in doc.ents:  # Named entities
                # if key.lower() in ent.text.lower():  # Check for a match with key
                extracted_value = ent.text
                break

            # Use extracted value if found; otherwise, use the default
            final_value = extracted_value if extracted_value is not None else default

            # Convert the value to the required data type
            try:
                final_value = safe_cast(final_value, data_type, default)
            except ValueError:
                raise ValueError(f"Could not convert {final_value} to {data_type}")

            # Store the result
            extracted_params[key] = final_value

        # Call the route's function with the input_text and extracted parameters
        return await route_info["function"](input_text, extracted_params)

    async def sbert_hnswlib(self, input_text: str):
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
            route_info = self.routes[most_similar_idx]
            return await self.execute_route(route_info, input_text)
        else:
            result = "No relevant match found."  # Default output

        # Output the result
        print("Input:", input_text)
        print("Similarity Score:", similarity_score)
        print("Result:", result)
        return result

    async def sbert_bm25_hybrid(self, input_text: str):
        """
        SBERT + BM25 Hybrid method for similarity search.
        """
        # Step 1: Use BM25 to prefilter the top candidates
        tokenized_query = input_text.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k BM25 candidates
        top_k_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:3]
        top_routes = [self.routes[i] for i in top_k_idx]
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

            print("Input:", input_text)
            print("BM25 Top Candidates:", top_routes)
            print("Max Cosine Similarity Score:", max_score)

            route_info = top_routes[best_match_idx]
            return await self.execute_route(route_info, input_text)
        else:
            result = "No relevant match found."  # Default output

        # Output the result
        print("Input:", input_text)
        print("BM25 Top Candidates:", top_routes)
        print("Max Cosine Similarity Score:", max_score)
        print("Result:", result)
        return result
