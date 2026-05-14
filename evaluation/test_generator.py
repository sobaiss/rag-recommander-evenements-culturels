# ruff: noqa: E402
# instructor (dépendance de ragas) fait `from mistralai import Mistral`.
# mistralai>=2.x est un namespace package sans __init__.py : Mistral est dans
# mistralai.client. Le shim ci-dessous doit précéder tout import de ragas.
import os
import sys
import warnings

# Ajoute la racine du projet au chemin pour résoudre `utils`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# llm_factory de ragas 0.4.3 ne supporte pas le SDK Mistral natif (attend
# messages.create style Anthropic). On conserve les wrappers LangChain et on
# supprime leurs warnings de dépréciation.
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=".*LangchainLLMWrapper.*"
)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=".*LangchainEmbeddingsWrapper.*"
)

# Charge le .env depuis la racine du projet (non chargé par uv run hors Makefile)
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import mistralai
import mistralai.client as _mc

mistralai.Mistral = _mc.Mistral  # type: ignore[attr-defined]

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from pydantic import SecretStr

# Imports depuis les modules internes pour contourner les DeprecationHelper de ragas :
# llm_factory ne supporte pas le SDK Mistral natif (ragas 0.4.3).
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.testset import Testset, TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.transforms import apply_transforms, default_transforms

from core.config import MISTRAL_API_KEY
from core.load_data import load_documents_from_file

current_path = os.curdir
data_dir = os.path.join(os.curdir, "..", "data")
report_dir = os.path.join(os.curdir, "..", "report")
os.makedirs(report_dir, exist_ok=True)
documents_filepath = os.path.join(data_dir, "eval_events.json")
documents = load_documents_from_file(documents_filepath)

kg = KnowledgeGraph()

for doc in documents:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            },
        )
    )


secretApiKey = SecretStr(MISTRAL_API_KEY if MISTRAL_API_KEY else "")

# 1 requête toutes les 3 s → ~20 RPM, en dessous des limites Mistral free tier.
# max_retries=6 déclenche un backoff exponentiel via tenacity sur les 429.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.33,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)
generator_llm = LangchainLLMWrapper(
    ChatMistralAI(
        api_key=secretApiKey,
        model_name="mistral-small-latest",
        temperature=0,
        max_retries=6,
        rate_limiter=rate_limiter,
    )
)
generator_embeddings = LangchainEmbeddingsWrapper(
    MistralAIEmbeddings(api_key=SecretStr(MISTRAL_API_KEY or ""), model="mistral-embed")
)

transformer_llm = generator_llm
embedding_model = generator_embeddings

trans = default_transforms(
    documents=documents, llm=transformer_llm, embedding_model=embedding_model
)
apply_transforms(kg, trans)

knowledge_graph_file = os.path.join(report_dir, "knowledge_graph.json")
kg.save(knowledge_graph_file)
loaded_kg = KnowledgeGraph.load(knowledge_graph_file)


generator = TestsetGenerator(
    llm=generator_llm,  # type: ignore[arg-type]
    embedding_model=embedding_model,
    knowledge_graph=loaded_kg,
)


query_distribution = default_query_distribution(generator_llm)

# Forcer la génération en français : on ajoute une directive à l'instruction
# de chaque synthesizer (single-hop et multi-hop ont tous generate_query_reference_prompt).
for synthesizer, _ in query_distribution:
    if hasattr(synthesizer, "generate_query_reference_prompt"):
        synthesizer.generate_query_reference_prompt.instruction += (  # type: ignore[attr-defined]
            "\nIMPORTANT: You MUST generate the query and the answer in French."
        )

testset = generator.generate(testset_size=5, query_distribution=query_distribution)
assert isinstance(testset, Testset)
df = testset.to_pandas()

testset_file = os.path.join(data_dir, "eval_dataset.json")
df.to_json(testset_file, orient="records", force_ascii=False, indent=2)
print(f"Testset saved to {testset_file} ({len(df)} samples)")
