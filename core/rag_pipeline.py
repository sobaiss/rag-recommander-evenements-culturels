import datetime
import logging
from dataclasses import dataclass, field

from mistralai.client import Mistral

from core.prompts import (
    rag_json_system_prompt,
    rag_no_results_system_prompt,
    rag_system_prompt,
)
from core.query_classifier import QueryClassifier
from core.vector_store import VectorStoreManager


def _is_ollama_model(model: str) -> bool:
    return model.startswith("ollama:")


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

    def __init__(
        self,
        query_classifier: QueryClassifier,
        vector_store: VectorStoreManager,
        mistral_client: Mistral | None,
    ) -> None:
        self.query_classifier = query_classifier
        self.vector_store = vector_store
        self.mistral_client = mistral_client

    def _call_llm(
        self,
        model: str,
        system_prompt: str,
        question: str,
        temperature: float,
        as_json: bool,
    ) -> str:
        if _is_ollama_model(model):
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_ollama import ChatOllama

            model_name = model.split(":", 1)[1]
            kwargs = {"format": "json"} if as_json else {}
            llm = ChatOllama(model=model_name, temperature=temperature, **kwargs)

            logging.info(
                f"Appel du modèle Ollama `{model_name}` avec température={temperature}..."
            )
            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=question),
                ]
            )
            return str(response.content)

        if self.mistral_client is None:
            raise ValueError(
                "MISTRAL_API_KEY non définie. Configurez-la dans .env ou utilisez un modèle Ollama."
            )
        kwargs = {"response_format": {"type": "json_object"}} if as_json else {}

        logging.info(
            f"Appel du modèle Mistral `{model}` avec température={temperature}..."
        )
        chat_response = self.mistral_client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=temperature,
            **kwargs,
        )
        message = chat_response.choices[0].message
        return str(message.content) if message is not None else ""

    def run(
        self,
        question: str,
        k: int = 5,
        min_score: float = 0.5,
        model: str = "mistral-large-latest",
        temperature: float = 0.1,
        as_json: bool = False,
    ) -> RAGResult:
        now = datetime.datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_month = now.strftime("%B %Y")

        # 1. Classification
        needs_rag, confidence, reason = self.query_classifier.needs_rag(question)
        mode = "RAG" if needs_rag else "DIRECT"
        logging.info(
            f"RAG pipeline — mode={mode} confiance={confidence:.2f} raison='{reason}'"
        )

        # 2. Recherche vectorielle (mode RAG uniquement)
        sources: list[dict] = []
        if needs_rag:
            retrieved = self.vector_store.search(question, k=k, min_score=min_score)
            sources = [
                {
                    "text": doc["text"],
                    "score": doc["score"],
                    "metadata": doc["metadata"],
                }
                for doc in retrieved
            ]

        # 3. Court-circuit si RAG sans résultat (pas d'appel LLM)
        if needs_rag and not sources:
            logging.warning(
                "Aucun document pertinent — réponse statique, appel LLM ignoré."
            )
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
            system_prompt = (
                rag_json_system_prompt(context_str, current_date, current_month)
                if as_json
                else rag_system_prompt(context_str, current_date, current_month)
            )
        else:
            system_prompt = rag_no_results_system_prompt(current_date)

        # 5. Génération de la réponse (exceptions propagées vers l'appelant)
        answer = self._call_llm(
            model=model,
            system_prompt=system_prompt,
            question=question,
            temperature=temperature,
            as_json=as_json and needs_rag,
        )
        logging.info(f"Réponse générée par {model}.")

        return RAGResult(
            answer=answer,
            mode=mode,
            confidence=confidence,
            reason=reason,
            sources=sources,
            model_used=model,
        )
