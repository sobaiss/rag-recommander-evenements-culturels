# ruff: noqa: E402
# Compatibility shim: instructor (dependency of ragas) does `from mistralai import Mistral`
# but mistralai 2.x is a namespace package with no top-level __init__.py, so that import fails.
# This must run before any ragas/instructor import.
import mistralai as _mistralai_ns
from mistralai.client import Mistral as _Mistral

if not hasattr(_mistralai_ns, "Mistral"):
    _mistralai_ns.Mistral = _Mistral

"""
evaluate_rag.py — Évaluation automatique du pipeline RAG avec Ragas.

Usage:
    uv run evaluate_rag.py [options]

    Options:
      --evaluator  mistral | ollama  (défaut: ollama)
      --ollama-model   modèle Ollama pour le LLM juge     (défaut: mistral)
      --ollama-embed   modèle Ollama pour les embeddings   (défaut: nomic-embed-text)
      --dataset    chemin vers le dataset annoté           (défaut: data/eval_dataset.json)
      --report     chemin du rapport de sortie             (défaut: data/eval_report.json)
      --k          nombre de documents à récupérer         (défaut: 5)

Exit codes:
    0 — tous les seuils sont atteints
    1 — au moins un seuil non atteint
    2 — erreur de configuration ou d'exécution

Prérequis Ollama:
    ollama pull mistral
    ollama pull nomic-embed-text
"""
import argparse
import json
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

THRESHOLDS = {
    "faithfulness": 0.5,
    "answer_relevancy": 0.5,
    "context_precision_with_reference": 0.4,
    "context_recall": 0.4,
}

EVAL_VECTOR_DB_DIR = "vector_db_eval"

from langchain_core.callbacks.base import BaseCallbackHandler

class DebugCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("=== PROMPTS ENVOYÉS À MISTRAL ===")
        for p in prompts:
            print(p)
            print("-" * 80)

def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_with_backoff(fn, max_retries: int = 6, base_wait: float = 5.0):
    """Appelle fn() et réessaie avec backoff exponentiel si une erreur 429 survient."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            is_429 = "429" in str(exc) or "Too Many Requests" in str(exc)
            if is_429 and attempt < max_retries - 1:
                wait = base_wait * (2 ** attempt)
                logging.warning(f"429 rate limit — attente {wait:.0f}s (tentative {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise


def run_pipeline(questions: list[str], k: int = 5, delay: float = 2.0) -> tuple[list[str], list[list[str]]]:
    """Run the RAG pipeline for each question and return answers + retrieved contexts."""
    try:
        from mistralai.client import Mistral

        from utils.config import MISTRAL_API_KEY
        from utils.query_classifier import QueryClassifier
        from utils.rag_pipeline import RAGPipeline
        from utils.vector_store import VectorStoreManager
    except ImportError as exc:
        logging.error(f"Import error: {exc}")
        sys.exit(2)

    if not MISTRAL_API_KEY:
        logging.error("MISTRAL_API_KEY non configurée.")
        sys.exit(2)

    vector_store = VectorStoreManager(vector_db_dir=EVAL_VECTOR_DB_DIR)
    if vector_store.index is None or vector_store.index.ntotal == 0:
        logging.error("L'index FAISS est vide. Lancez d'abord : make eval-build")
        sys.exit(2)

    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    classifier = QueryClassifier()
    pipeline = RAGPipeline(classifier, vector_store, mistral_client)

    answers: list[str] = []
    contexts: list[list[str]] = []

    for i, question in enumerate(questions):
        if i > 0 and delay > 0:
            time.sleep(delay)
        logging.info(f"Question {i + 1}/{len(questions)} : {question!r}")
        result = _run_with_backoff(
            lambda q=question: pipeline.run(question=q, k=k, min_score=0.0, model="mistral-small-latest")
        )
        answers.append(result.answer)
        contexts.append([s["text"] for s in result.sources] if result.sources else [""])

    return answers, contexts


def build_ragas_dataset(questions, ground_truths, answers, contexts):
    try:
        from ragas import EvaluationDataset, SingleTurnSample
    except ImportError:
        logging.error("Package 'ragas' manquant. Installez avec : uv sync")
        sys.exit(2)

    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=ctx,
            reference=gt,
        )
        for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths)
    ]
    return EvaluationDataset(samples=samples)


def _build_evaluator_ollama(ollama_model: str, ollama_embed: str):
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from langchain_ollama import ChatOllama, OllamaEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        logging.error(f"Package manquant ({exc}). Installez avec : uv add langchain-ollama")
        sys.exit(2)

    logging.info(f"Évaluateur : Ollama ({ollama_model} + {ollama_embed})")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return (
            LangchainLLMWrapper(ChatOllama(model=ollama_model, temperature=0)),
            LangchainEmbeddingsWrapper(OllamaEmbeddings(model=ollama_embed)),
            None,
        )


def _build_evaluator_mistral():
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from langchain_mistralai.chat_models import ChatMistralAI
            from langchain_mistralai.embeddings import MistralAIEmbeddings
            from ragas import RunConfig
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        logging.error(f"Package Ragas manquant ({exc}). Installez avec : uv sync")
        sys.exit(2)

    from utils.config import MISTRAL_API_KEY

    logging.info("Évaluateur : Mistral API (avec retry sur 429)")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return (
            LangchainLLMWrapper(
                ChatMistralAI(mistral_api_key=MISTRAL_API_KEY, model="mistral-small-latest", temperature=0)
            ),
            LangchainEmbeddingsWrapper(
                MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY, model="mistral-embed")
            ),
            RunConfig(max_retries=5, max_wait=60),
        )


def evaluate(dataset, evaluator: str, ollama_model: str, ollama_embed: str) -> dict[str, float]:
    import warnings

    try:
        # ragas.metrics.collections hérite de BaseMetric (nouveau) mais evaluate() valide
        # contre l'ancienne classe Metric — bug ragas 0.4.3. On utilise ragas.metrics (ancien
        # chemin) qui passe le check, en supprimant les DeprecationWarnings associés.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                LLMContextPrecisionWithReference,
                LLMContextRecall,
                answer_relevancy,
                faithfulness,
            )
    except ImportError as exc:
        logging.error(f"Package Ragas manquant ({exc}). Installez avec : uv sync")
        sys.exit(2)

    if evaluator == "ollama":
        llm, embeddings, run_config = _build_evaluator_ollama(ollama_model, ollama_embed)
    else:
        llm, embeddings, run_config = _build_evaluator_mistral()

    metrics = [
        faithfulness,
        answer_relevancy,
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]
    faithfulness.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings

    kwargs = {"dataset": dataset, "metrics": metrics, "callbacks": [DebugCallback()]}
    if run_config is not None:
        kwargs["run_config"] = run_config

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
        result = ragas_evaluate(**kwargs)

    df = result.to_pandas()
    return {col: float(df[col].mean()) for col in THRESHOLDS if col in df.columns}


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation RAG avec Ragas")
    parser.add_argument("--dataset", default="data/eval_dataset.json")
    parser.add_argument("--report", default="data/eval_report.json")
    parser.add_argument("--k", type=int, default=5, help="Nombre de documents à récupérer")
    parser.add_argument(
        "--pipeline-delay",
        type=float,
        default=2.0,
        help="Délai en secondes entre chaque appel pipeline (évite le 429 Mistral, défaut: 2s)",
    )
    parser.add_argument(
        "--evaluator",
        choices=["ollama", "mistral"],
        default="ollama",
        help="LLM juge pour Ragas : ollama (local, défaut) ou mistral (API)",
    )
    parser.add_argument("--ollama-model", default="mistral", help="Modèle Ollama pour le LLM juge")
    parser.add_argument(
        "--ollama-embed", default="nomic-embed-text", help="Modèle Ollama pour les embeddings"
    )
    args = parser.parse_args()

    try:
        pairs = load_dataset(args.dataset)
    except FileNotFoundError:
        logging.error(f"Dataset introuvable : {args.dataset}")
        sys.exit(2)

    questions = [p["question"] for p in pairs]
    ground_truths = [p["ground_truth"] for p in pairs]
    answers = [p["answer"] for p in pairs]
    contexts = [p["contexts"] for p in pairs]

    # answers, contexts = run_pipeline(questions, k=args.k, delay=args.pipeline_delay)

    ragas_dataset = build_ragas_dataset(questions, ground_truths, answers, contexts)

    logging.info("Lancement de l'évaluation Ragas...")
    scores = evaluate(ragas_dataset, args.evaluator, args.ollama_model, args.ollama_embed)

    passed = {metric: score >= THRESHOLDS[metric] for metric, score in scores.items()}
    report = {
        "scores": scores,
        "thresholds": THRESHOLDS,
        "passed": passed,
        "all_passed": all(passed.values()),
        "evaluator": args.evaluator,
        "qa_pairs": [
            {"question": q, "answer": a, "ground_truth": gt}
            for q, a, gt in zip(questions, answers, ground_truths)
        ],
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logging.info("─── Résultats Ragas ───────────────────────────────")
    for metric, score in scores.items():
        status = "✓" if passed[metric] else "✗"
        logging.info(f"  {status} {metric}: {score:.3f}  (seuil: {THRESHOLDS[metric]})")
    logging.info(f"Rapport sauvegardé : {args.report}")

    sys.exit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
