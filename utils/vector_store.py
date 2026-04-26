import logging
import os
import pickle

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai.client import Mistral

from utils.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_CHUNKS_FILE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    MISTRAL_API_KEY,
)


class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss."""
    def __init__(self):
        self.index: faiss.Index | None = None  # L'instance de l'index Faiss
        self.document_chunks: list[dict[str, any]] = []
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        self._load_index_and_chunks()

    def _load_index_and_chunks(self):
        """Charge l'index Faiss et les chunks de documents depuis le disque, ou initialise des structures vides si aucun index n'existe."""
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE):
            try:
                logging.info(f"Chargement de l'index Faiss depuis {FAISS_INDEX_FILE}.")
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                logging.info(f"Chargement des chunks de documents depuis {DOCUMENT_CHUNKS_FILE}.")
                with open(DOCUMENT_CHUNKS_FILE, 'rb') as f:
                    self.document_chunks = pickle.load(f)
                logging.info(f"Index ({self.index.ntotal} vecteurs) et {len(self.document_chunks)} chunks chargés.")
            except Exception as e:
                logging.error(f"Erreur lors du chargement de l'index/chunks Faiss: {e}")
                self.index = None
                self.document_chunks = []
        else:
            self.index = None
            self.document_chunks = []
            logging.info("Aucun index existant trouvé. Initialisation d'un nouvel index vide.")

    def build_index(self, documents: list[Document]):
        """Construit l'index Faiss à partir des documents."""
        if not documents:
            logging.warning("Aucun document fourni pour la construction de l'index.")
            return
        # 1. Découper en chunks
        logging.info("Découpage des documents en chunks...")
        self.document_chunks = self._split_documents_to_chunks(documents)
        if not self.document_chunks:
            logging.warning("Aucun chunk généré à partir des documents fournis.")
            return
        # 2. Générer les embeddings
        logging.info("Génération des embeddings pour les chunks...")
        embeddings = self._generate_embeddings(self.document_chunks)
        if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
            logging.error("Échec de la génération des embeddings. L'index ne sera pas construit.")
            #Nettoyer les données partiellement générées pour éviter des incohérences
            self.document_chunks = []
            self.index = None
            #Supprimer les fichiers d'index et de chunks partiellement créés
            if os.path.exists(FAISS_INDEX_FILE):
                os.remove(FAISS_INDEX_FILE)
            if os.path.exists(DOCUMENT_CHUNKS_FILE):
                os.remove(DOCUMENT_CHUNKS_FILE)
            return

        # 3. Créer l'index Faiss optimisé pour la recherche de similarité cosinus
        logging.info("Création de l'index Faiss...")
        dimension = embeddings.shape[1]
        # Normaliser les embeddings pour la similarité cosinus
        faiss.normalize_L2(embeddings)

        # Utiliser IndexFlatIP pour la similarité cosinus (IndexFlatIP = produit scalaire)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logging.info(f"Index Faiss créé avec {self.index.ntotal} vecteurs.")

        # 4. Sauvegarder l'index et les chunks sur le disque
        self._save_index_and_chunks()

    def _save_index_and_chunks(self):
        """Sauvegarde l'index Faiss et la liste des chunks."""
        if self.index is None or not self.document_chunks:
            logging.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return

        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_FILE), exist_ok=True)

        try:
            logging.info(f"Sauvegarde de l'index Faiss dans {FAISS_INDEX_FILE}...")
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            logging.info(f"Sauvegarde des chunks dans {DOCUMENT_CHUNKS_FILE}...")
            with open(DOCUMENT_CHUNKS_FILE, "wb") as f:
                pickle.dump(self.document_chunks, f)
            logging.info("Index et chunks sauvegardés avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de l'index/chunks: {e}")

    def _split_documents_to_chunks(self, documents: list[Document]) -> list[Document]:
        """Découpe les documents en chunks selon la configuration définie."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len, # Important: mesure en caractères
            add_start_index=True, # Ajoute la position de début du chunk dans le document original
        )

        all_chunks = []
        doc_counter = 0
        for doc in documents:
            chunks = text_splitter.split_documents([doc])
            logging.info(
                f"  Document '{doc.metadata.get('uid', 'N/A')}' découpé en {len(chunks)} chunks."
            )
            for i, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "id": f"doc{doc_counter}_{i}",  # ID unique pour chaque chunk
                        "text": chunk.page_content,
                        "metadata": {
                            **chunk.metadata,
                            "chunk_id_in_doc": i,
                            "start_index": chunk.metadata.get(
                                "start_index", -1
                            ),  # Position de début (en caractères)
                        },
                    }
                )
            doc_counter += 1
        return all_chunks


    def _generate_embeddings(self, chunks: list[dict[str, any]]) -> np.ndarray | None:
        """Génère les embeddings pour une liste de chunks de documents."""
        if not MISTRAL_API_KEY:
            logging.error("Impossible de générer les embeddings: MISTRAL_API_KEY manquante.")
            return None
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        logging.info(
            f"Génération des embeddings pour {len(chunks)} chunks (modèle: {EMBEDDING_MODEL})..."
        )

        all_embeddings = []
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
            texts_to_embed = [chunk["text"] for chunk in batch_chunks]

            logging.info(
                f"  Traitement du lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)"
            )

            try:
                response = self.mistral_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    inputs=texts_to_embed
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logging.error(
                    f"Erreur inattendue lors de la génération d'embeddings (lot {batch_num}): {e}"
                )
                # Gérer comme ci-dessus
                num_failed = len(texts_to_embed)
                if all_embeddings:
                    dim = len(all_embeddings[0])
                else:
                    logging.error(
                        "Impossible de déterminer la dimension des embeddings, saut du lot."
                    )
                    continue
                logging.warning(
                    f"Ajout de {num_failed} vecteurs nuls de dimension {dim} pour le lot échoué."
                )
                all_embeddings.extend([np.zeros(dim, dtype="float32")] * num_failed)

        if not all_embeddings:
            logging.error("Aucun embedding généré avec succès.")
            return None

        embeddings_array =  np.array(all_embeddings, dtype="float32")
        logging.info(f"Génération des embeddings terminée. Forme finale: {embeddings_array.shape}")
        return embeddings_array

    def search(self, query_text: str, k: int = 5, min_score: float = None) -> list[dict[str, any]]:
        """
        Recherche les k chunks les plus pertinents pour une requête.

        Args:
            query_text: Texte de la requête
            k: Nombre de résultats à retourner
            min_score: Score minimum (entre 0 et 1) pour inclure un résultat

        Returns:
            Liste des chunks pertinents avec leurs scores
        """
        if self.index is None or not self.document_chunks:
            logging.warning("Recherche impossible: l'index Faiss n'est pas chargé ou est vide.")
            return []
        if not MISTRAL_API_KEY:
            logging.error(
                "Recherche impossible: MISTRAL_API_KEY manquante pour générer l'embedding de la requête."
            )
            return []

        logging.info(f"Recherche des {k} chunks les plus pertinents pour: '{query_text}'")
        try:
            # 1. Générer l'embedding de la requête
            response = self.mistral_client.embeddings.create(
                model=EMBEDDING_MODEL,
                inputs=[query_text],
            )
            query_embedding = np.array([response.data[0].embedding]).astype("float32")

            # Normaliser l'embedding de la requête pour la similarité cosinus
            faiss.normalize_L2(query_embedding)

            # 2. Rechercher dans l'index Faiss
            # Pour IndexFlatIP: scores = produit scalaire (plus grand = meilleur)
            # indices: index des chunks correspondants dans self.document_chunks
            # Demander plus de résultats si un score minimum est spécifié
            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_embedding, search_k)

            # 3. Formater les résultats
            results = []
            if indices.size > 0:  # Vérifier s'il y a des résultats
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.document_chunks):  # Vérifier la validité de l'index
                        chunk = self.document_chunks[idx]
                        # Convertir le score en similarité (0-1)
                        # Pour IndexFlatIP avec vecteurs normalisés, le score est déjà entre -1 et 1
                        # On le convertit en pourcentage (0-100%)
                        raw_score = float(scores[0][i])
                        similarity = raw_score * 100

                        # Filtrer les résultats en fonction du score minimum
                        # Le min_score est entre 0 et 1, mais similarity est en pourcentage (0-100)
                        min_score_percent = min_score * 100 if min_score is not None else 0
                        if min_score is not None and similarity < min_score_percent:
                            logging.debug(
                                f"Document filtré (score {similarity:.2f}% < minimum {min_score_percent:.2f}%)"
                            )
                            continue

                        results.append(
                            {
                                "score": similarity,  # Score de similarité en pourcentage
                                "raw_score": raw_score,  # Score brut pour débogage
                                "text": chunk["text"],
                                "metadata": chunk[
                                    "metadata"
                                ],  # Contient source, category, chunk_id_in_doc, start_index etc.
                            }
                        )
                    else:
                        logging.warning(
                            f"Index Faiss {idx} hors limites (taille des chunks: {len(self.document_chunks)})."
                        )

            # Trier par score (similarité la plus élevée en premier)
            results.sort(key=lambda x: x["score"], reverse=True)

            # Limiter au nombre demandé (k) si nécessaire
            if len(results) > k:
                results = results[:k]

            if min_score is not None:
                min_score_percent = min_score * 100
                logging.info(
                    f"{len(results)} chunks pertinents trouvés (score minimum: {min_score_percent:.2f}%)."
                )
            else:
                logging.info(f"{len(results)} chunks pertinents trouvés.")

            return results

        # except MistralAPIException as e:
        #     logging.error(
        #         f"Erreur API Mistral lors de la génération de l'embedding de la requête: {e}"
        #     )
        #     logging.error(f"  Détails: Status Code={e.status_code}, Message={e.message}")
        #     return []
        except Exception as e:
            logging.error(f"Erreur inattendue lors de la recherche: {e}")
            return []

    def clear_index(self):
        """Supprime l'index Faiss et les chunks de documents du disque et réinitialise les structures en mémoire."""
        self.index = None
        self.document_chunks = []
        if os.path.exists(FAISS_INDEX_FILE):
            os.remove(FAISS_INDEX_FILE)
            logging.info(f"Fichier d'index Faiss supprimé: {FAISS_INDEX_FILE}")
        if os.path.exists(DOCUMENT_CHUNKS_FILE):
            os.remove(DOCUMENT_CHUNKS_FILE)
            logging.info(f"Fichier de chunks supprimé: {DOCUMENT_CHUNKS_FILE}")
        logging.info("Index et chunks réinitialisés.")
