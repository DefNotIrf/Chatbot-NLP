import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self, csv_path):
        # 'all-MiniLM-L6-v2' is the best balance of speed/accuracy for Pi
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2') 
        self.df = pd.read_csv(csv_path)
        self.index = None
        self._build_index()

    def _build_index(self):
        self.df['text_repr'] = self.df.apply(
            lambda x: (
                f"Stall: {x['stall_name']}. "
                f"Meal: {x['meal_name']}. "
                f"Tags: {str(x['tags']).replace(';', ', ')}. "
                f"Price: {x['price']} MYR. "
                f"Time: {x['available_time']}"
            ),
            axis=1
        )
        
        # Vectorize
        embeddings = self.encoder.encode(self.df['text_repr'].tolist())
        dimension = embeddings.shape[1]
        
        # Build FAISS Index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query, k=20):
        # Semantic search
        query_vec = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.df.iloc[idx]['text_repr'])
        return "\n".join(results)
