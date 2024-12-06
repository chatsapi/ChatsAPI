import hnswlib
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi


class ChatsAPI:
    def __init__(self, routes: list[str]):
        self.routes = routes

        # Initialize SBERT model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Step 1: Precompute embeddings for all items (for SBERT + HNSWlib)
        self.embeddings = self.model.encode(self.routes, normalize_embeddings=True)

        # Initialize HNSWlib index
        dim = self.embeddings.shape[1]
        self.hnsw_index = hnswlib.Index(space='cosine', dim=dim)
        self.hnsw_index.init_index(max_elements=len(self.embeddings), ef_construction=100, M=16)
        self.hnsw_index.add_items(self.embeddings)

        # Initialize BM25 (for SBERT + BM25 Hybrid Approach)
        self.tokenized_routes = [route.split() for route in self.routes]
        self.bm25 = BM25Okapi(self.tokenized_routes)

    def chat(self, input_text: str, method="hnswlib"):
        """
        Handles chat queries using the specified method: 'hnswlib' or 'bm25_hybrid'.

        :param input_text: User query
        :param method: Search method ('hnswlib' or 'bm25_hybrid')
        :return: The most similar route or a default response
        """

        if method == "hnswlib":
            return self.sbert_hnswlib(input_text)
        elif method == "bm25_hybrid":
            return self.sbert_bm25_hybrid(input_text)
        else:
            raise ValueError("Invalid method. Choose 'hnswlib' or 'bm25_hybrid'.")

    def sbert_hnswlib(self, input_text: str):
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
            result = self.routes[most_similar_idx]
        else:
            result = "No relevant match found."  # Default output

        # Output the result
        print("Input:", input_text)
        print("Similarity Score:", similarity_score)
        print("Result:", result)
        return result

    def sbert_bm25_hybrid(self, input_text: str):
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
            result = top_routes[best_match_idx]
        else:
            result = "No relevant match found."  # Default output

        # Output the result
        print("Input:", input_text)
        print("BM25 Top Candidates:", top_routes)
        print("Max Cosine Similarity Score:", max_score)
        print("Result:", result)
        return result
