import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class QASystem:
    
    def __init__(self, 
                 embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        print(f"Cargando modelo de embeddings: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self.index = None
        self.documents = []
        self.metadata = []
            
    def load_data(self, csv_path: str) -> pd.DataFrame:
        print(f"Cargando datos de {csv_path}")
        encodings = ['utf-8', 'latin-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                print(f"Datos cargados exitosamente")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("No se pudo cargar el CSV")
        
        df.columns = df.columns.str.strip()
        
        print(f"Cargados {len(df)} registros")
        return df
    
    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if end < len(text):
                last_period = chunk.rfind('.')
                last_comma = chunk.rfind(',')
                last_space = chunk.rfind(' ')
                
                cut_point = max(last_period, last_comma, last_space)
                if cut_point > chunk_size * 0.5:
                    chunk = chunk[:cut_point + 1]
                    end = start + cut_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def prepare_documents(self, df: pd.DataFrame, chunk_documents: bool = True):
        print("Preparando documentos...")
        self.documents = []
        self.metadata = []
        
        for _, row in df.iterrows():
            query = str(row.get('customer_query', ''))
            response = str(row.get('support_response', ''))
            
            document = f"Pregunta: {query}\nRespuesta: {response}"
            
            if chunk_documents and len(document) > 300:
                chunks = self.chunk_text(document, chunk_size=300, overlap=50)
                for chunk in chunks:
                    self.documents.append(chunk)
                    self.metadata.append({
                        'id': row.get('id'),
                        'category': row.get('category'),
                        'original_query': query,
                        'original_response': response,
                        'created_at': row.get('created_at')
                    })
            else:
                self.documents.append(document)
                self.metadata.append({
                    'id': row.get('id'),
                    'category': row.get('category'),
                    'original_query': query,
                    'original_response': response,
                    'created_at': row.get('created_at')
                })
        
        print(f"Total de documentos preparados: {len(self.documents)}")
    
    def generate_embeddings(self):
        print("Generando embeddings...")
        embeddings = self.embedding_model.encode(
            self.documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Índice FAISS creado con {self.index.ntotal} vectores")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.index is None:
            raise ValueError("El índice no ha sido creado. Ejecuta generate_embeddings() primero.")
        
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(distance),
                    'rank': i + 1
                })
        
        return results
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        return self._generate_simple_answer(query, context_docs)
    
    def _generate_simple_answer(self, query: str, context_docs: List[Dict]) -> str:
        if not context_docs:
            return "Lo siento, no encontré información relevante para tu pregunta."
        
        best_doc = context_docs[0]
        metadata = best_doc['metadata']
        
        document = best_doc['document']
        if "Respuesta:" in document:
            answer = document.split("Respuesta:")[-1].strip()
        else:
            answer = metadata.get('original_response', document)
        
        if len(context_docs) > 1:
            answer += f"\n\n(Información relacionada encontrada en {len(context_docs)} documentos)"
        
        return answer
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        search_results = self.search(question, top_k=top_k)
        
        answer = self.generate_answer(question, search_results)
        
        return {
            'question': question,
            'answer': answer,
            'sources': search_results,
            'num_sources': len(search_results)
        }
    
    def save(self, save_dir: str = "./qa_system_data"):
        os.makedirs(save_dir, exist_ok=True)
        
        faiss.write_index(self.index, os.path.join(save_dir, "faiss.index"))
        
        with open(os.path.join(save_dir, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(save_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Sistema guardado en {save_dir}")
    
    def load(self, save_dir: str = "./qa_system_data"):
        self.index = faiss.read_index(os.path.join(save_dir, "faiss.index"))
        
        with open(os.path.join(save_dir, "documents.pkl"), 'rb') as f:
            self.documents = pickle.load(f)
        
        with open(os.path.join(save_dir, "metadata.pkl"), 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Sistema cargado desde {save_dir}")


def main():
    qa = QASystem()
    
    df = qa.load_data("data/data.csv")
    
    qa.prepare_documents(df, chunk_documents=True)
    
    qa.generate_embeddings()
    
    qa.save()
    
    print("\n" + "="*60)
    print("SISTEMA DE QUESTION ANSWERING")
    print("="*60 + "\n")
    
    test_queries = [
        "¿Cómo cambio mi dirección?",
        "Mi tarjeta está bloqueada",
        "¿Cuánto tarda un reclamo?",
        "¿Cómo compro dólares?"
    ]
    
    for query in test_queries:
        print(f"Pregunta: {query}")
        print("-" * 60)
        result = qa.query(query, top_k=2)
        print(f"Respuesta: {result['answer']}")
        print(f"\nFuentes encontradas: {result['num_sources']}")
        for i, source in enumerate(result['sources'][:2], 1):
            print(f"  {i}. Categoría: {source['metadata']['category']} (score: {source['score']:.4f})")
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()