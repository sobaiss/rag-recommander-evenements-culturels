import datetime
import logging
from dataclasses import dataclass, field

from utils.prompts import rag_no_results_system_prompt, rag_system_prompt

NO_RESULTS_ANSWER = (
    "Je n'ai trouvé aucun événement correspondant à votre recherche dans la base indexée.\n\n"
    "Vous pouvez :\n"
    "- **Reformuler** votre question en utilisant d'autres mots-clés.\n"
    "- **Élargir les critères** (ville, période, thématique).\n"
    "- **Réindexer** la base de données avec d'autres filtres."
)


@dataclass
class RAGResult:
    answer: str
    mode: str
    confidence: float
    reason: str
    sources: list[dict] = field(default_factory=list)
    model_used: str = ""


class RAGPipeline:
    """Encapsule le flux classify → search → prompt → LLM commun à Chat.py et à l'API."""

    def __init__(self, query_classifier, vector_store, mistral_client):
        self.query_classifier = query_classifier
        self.vector_store = vector_store
        self.mistral_client = mistral_client

    def run(
        self,
        question: str,
        k: int = 5,
        min_score: float = 0.75,
        model: str = "mistral-large-latest",
        temperature: float = 0.1,
    ) -> RAGResult:
        now = datetime.datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_month = now.strftime("%B %Y")

        # 1. Classification
        needs_rag, confidence, reason = self.query_classifier.needs_rag(question)
        mode = "RAG" if needs_rag else "DIRECT"
        logging.info(f"RAG pipeline — mode={mode} confiance={confidence:.2f} raison='{reason}'")

        # 2. Recherche vectorielle (mode RAG uniquement)
        sources: list[dict] = []
        if needs_rag:
            retrieved = self.vector_store.search(question, k=k, min_score=min_score)
            sources = [
                {"text": doc["text"], "score": doc["score"], "metadata": doc["metadata"]}
                for doc in retrieved
            ]

        # 3. Court-circuit si RAG sans résultat (pas d'appel LLM)
        if needs_rag and not sources:
            logging.warning("Aucun document pertinent — réponse statique, appel LLM ignoré.")
            return RAGResult(
                answer=NO_RESULTS_ANSWER,
                mode=mode,
                confidence=confidence,
                reason=reason,
                sources=[],
                model_used=model,
            )

        # 4. Construction du prompt système
        if needs_rag:
            context_str = "\n\n---\n\n".join(
                f"Source: {s['metadata'].get('source', 'Inconnue')} (Score: {s['score']:.2f}%)\nContenu: {s['text']}"
                for s in sources
            )
            system_prompt = rag_system_prompt(context_str, current_date, current_month)
        else:
            system_prompt = rag_no_results_system_prompt(current_date)

        # 5. Génération de la réponse (exceptions propagées vers l'appelant)
        chat_response = self.mistral_client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=temperature,
        )
        answer = chat_response.choices[0].message.content
        logging.info("Réponse générée par Mistral.")

        return RAGResult(
            answer=answer,
            mode=mode,
            confidence=confidence,
            reason=reason,
            sources=sources,
            model_used=model,
        )
