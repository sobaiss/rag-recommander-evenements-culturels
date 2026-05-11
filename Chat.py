# app.py
import datetime
import logging
import os

import streamlit as st
from mistralai.client import Mistral
from streamlit_feedback import streamlit_feedback  # Importez le composant

# Importer nos modules locaux
from utils.config import (
    APP_TITLE,
    AVAILABLE_EMBEDDING_MODELS,
    COMPANY_NAME,
    DOCUMENT_CHUNKS_FILE,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    INDEX_METADATA_FILE,
    MISTRAL_API_KEY,
    FRENCH_CITIES,
)
from utils.database import log_interaction, update_feedback  # Importez update_feedback
from utils.load_data import build_openagenda_url, load_documents_from_url_paginated, save_documents_to_json
from utils.query_classifier import QueryClassifier
from utils.rag_pipeline import RAGPipeline
from utils.vector_store import VectorStoreManager

class StreamlitLogHandler(logging.Handler):
    """Capture les messages logging Python pour les afficher dans Streamlit."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []
        self.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration de la page Streamlit ---
st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

# --- Initialisation (avec mise en cache Streamlit) ---


# Met en cache le VectorStoreManager pour éviter de recharger l'index à chaque interaction
@st.cache_resource
def get_vector_store():
    logging.info("Chargement du VectorStoreManager...")
    return VectorStoreManager()


# Met en cache le client Mistral
@st.cache_resource
def get_mistral_client():
    if not MISTRAL_API_KEY:
        st.error("Erreur: La clé API Mistral (MISTRAL_API_KEY) n'est pas configurée.")
        st.stop()
    logging.info("Initialisation du client Mistral...")
    return Mistral(api_key=MISTRAL_API_KEY)


# Met en cache le classificateur de requêtes
@st.cache_resource
def get_query_classifier():
    logging.info("Initialisation du classificateur de requêtes...")
    return QueryClassifier()


# Charge le Vector Store, le client Mistral et le classificateur de requêtes
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vector_store()
vector_store = st.session_state.vector_store
client = get_mistral_client()
query_classifier = get_query_classifier()

# Initialise l'historique du chat dans l'état de la session s'il n'existe pas
if "messages" not in st.session_state:
    st.session_state.messages = []
# Initialise l'ID de la dernière interaction pour le feedback
if "last_interaction_id" not in st.session_state:
    st.session_state.last_interaction_id = None
# Contrôle l'affichage du formulaire de réindexation
if "show_reindex_form" not in st.session_state:
    st.session_state.show_reindex_form = False

# --- Interface Utilisateur ---

# Barre latérale (sidebar)
with st.sidebar:
    st.title(f"📚 {COMPANY_NAME}")
    st.caption(f"{APP_TITLE}")

    # Bouton pour lancer une nouvelle conversation
    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        # Réinitialiser l'historique des messages
        st.session_state.messages = []
        st.session_state.last_interaction_id = None
        st.rerun()  # Recharger l'application pour afficher la nouvelle conversation

    st.divider()

    # Paramètres de l'application
    st.subheader("⚙️ Paramètres")

    # Sélecteur de modèle Mistral
    model_options = {
        "mistral-small-latest": "Mistral Small (rapide)",
        "mistral-large-latest": "Mistral Large (précis)",
    }
    selected_model = st.selectbox(
        "Modèle LLM",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=1,  # Large par défaut
    )

    # Slider pour le nombre de documents
    num_docs = st.slider(
        "Nombre de documents à récupérer",
        min_value=1,
        max_value=20,
        value=5,  # 5 par défaut
        step=1,
    )

    # Slider pour le score minimum (en pourcentage)
    min_score_percent = st.slider(
        "Score minimum (filtrer les résultats faibles)",
        min_value=0,
        max_value=100,
        value=75,  # 75% par défaut
        step=5,
        format="%d%%",
    )
    # Convertir le pourcentage en valeur décimale (0-1)
    min_score = min_score_percent / 100.0

    st.divider()

    # Informations sur l'application
    st.subheader("📝 Informations")
    st.markdown(f"**Modèle sélectionné**: {model_options[selected_model]}")
    st.markdown(f"**Documents indexés**: {vector_store.index.ntotal if vector_store.index else 0}")

    # Informations sur la conversation actuelle
    if st.session_state.messages:
        st.info(f"{len(st.session_state.messages) // 2} échanges dans cette conversation")

        # Bouton pour télécharger la conversation
        # Préparer le contenu de la conversation au format texte
        conversation_text = "\n\n".join(
            [
                f"{'Utilisateur' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in st.session_state.messages
            ]
        )

        # Ajouter un en-tête avec la date et le titre
        header = f"Conversation avec l'assistant virtuel de {COMPANY_NAME}\n"
        header += f"Date: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
        conversation_text = header + conversation_text

        # Bouton de téléchargement
        st.download_button(
            label="💾 Télécharger la conversation",
            data=conversation_text,
            file_name=f"conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()

    # --- Section réindexation ---
    st.subheader("🗄️ Index vectoriel")
    current_meta = vector_store.get_metadata()
    if current_meta:
        st.caption(f"Modèle : `{current_meta.get('embedding_model', 'N/A')}`")
        st.caption(f"Chunks : {current_meta.get('num_chunks', 0)} · Docs : {current_meta.get('num_documents', 0)}")
        created = current_meta.get("created_at", "")[:10]
        if created:
            st.caption(f"Créé le : {created}")
    else:
        st.caption("Aucun index chargé.")

    reindex_label = "🔼 Masquer la réindexation" if st.session_state.show_reindex_form else "🔄 Réindexer la base"
    if st.button(reindex_label, use_container_width=True):
        st.session_state.show_reindex_form = not st.session_state.show_reindex_form
        st.rerun()

# --- Formulaire de réindexation (zone principale) ---
if st.session_state.show_reindex_form:
    st.header("🗄️ Réindexation de la base de connaissances")
    st.caption("Configurez les paramètres puis lancez la réindexation depuis l'API OpenAgenda.")

    current_meta = vector_store.get_metadata()
    default_model = (current_meta or {}).get("embedding_model", EMBEDDING_MODEL)

    model_keys = list(AVAILABLE_EMBEDDING_MODELS.keys())
    default_model_idx = model_keys.index(default_model) if default_model in model_keys else 0

    with st.form("reindex_form"):
        col1, col2 = st.columns(2)

        with col1:
            embedding_model = st.selectbox(
                "Modèle d'embedding *",
                options=model_keys,
                format_func=lambda x: AVAILABLE_EMBEDDING_MODELS[x],
                index=default_model_idx,
                help="mistral-embed utilise l'API Mistral. Les autres modèles sont locaux (nécessitent sentence-transformers).",
            )

        with col2:
            _one_year_ago = datetime.date.today() - datetime.timedelta(days=365)
            begin_date = st.date_input(
                "Date de début des événements",
                value=_one_year_ago,
                min_value=_one_year_ago,
                help="Filtre les événements à partir de ce mois. La date minimale est aujourd'hui − 1 an.",
            )
            locations = st.multiselect(
                "Villes",
                options=FRENCH_CITIES,
                default=["Paris"],
                help="Sélectionnez une ou plusieurs villes. Laissez vide pour toute la France.",
            )
            max_records = st.slider(
                "Nombre maximum de documents",
                min_value=10,
                max_value=100000,
                value=1000,
                step=100,
                help="Limite le nombre d'événements récupérés depuis l'API.",
            )

        submitted = st.form_submit_button("🚀 Lancer la réindexation", use_container_width=True, type="primary")

    if submitted:
        # --- Validation ---
        errors = []
        if not begin_date and not locations:
            errors.append("Veuillez sélectionner au moins une région ou une date de début.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            _log_handler = StreamlitLogHandler()
            logging.getLogger().addHandler(_log_handler)

            with st.status("Réindexation en cours...", expanded=True) as reindex_status:
                try:
                    # 1. Construction de l'URL
                    st.write("🔗 **Étape 1/5** — Construction de l'URL OpenAgenda...")
                    url = build_openagenda_url(locations, begin_date)
                    st.code(url, language=None)

                    # 2. Récupération des données
                    st.write("📥 **Étape 2/5** — Récupération des données (avec pagination)...")
                    documents = load_documents_from_url_paginated(url, max_records=max_records)
                    if not documents:
                        reindex_status.update(label="❌ Aucun événement trouvé", state="error")
                        st.error("Aucun événement ne correspond aux critères sélectionnés. Essayez d'autres filtres.")
                        logging.getLogger().removeHandler(_log_handler)
                        st.stop()
                    st.write(f"✅ **{len(documents)} événements** récupérés depuis l'API.")

                    # 3. Sauvegarde dans data/
                    st.write("💾 **Étape 3/5** — Sauvegarde des données dans `data/`...")
                    save_path = save_documents_to_json(documents)
                    st.write(f"✅ Fichier sauvegardé : `{save_path}`")

                    # 4. Suppression de l'ancien index
                    st.write("🗑️ **Étape 4/5** — Suppression de l'ancien index FAISS...")
                    for path in [FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE, INDEX_METADATA_FILE]:
                        if os.path.exists(path):
                            os.remove(path)
                            st.write(f"   Supprimé : `{path}`")
                    st.write("✅ Ancien index supprimé.")

                    # 5. Construction du nouvel index
                    st.write(f"🔨 **Étape 5/5** — Construction de l'index avec `{embedding_model}`...")
                    if not embedding_model.startswith("mistral"):
                        st.write(
                            "📦 Chargement du modèle HuggingFace "
                            "(première utilisation = téléchargement, cela peut prendre quelques minutes)..."
                        )

                    new_store = VectorStoreManager(embedding_model=embedding_model)

                    def _progress(msg: str) -> None:
                        st.write(f"   → {msg}")

                    new_store.build_index(documents, progress_callback=_progress)

                    # Rechargement du cache Streamlit + mise à jour immédiate du store actif
                    get_vector_store.clear()
                    st.session_state.vector_store = new_store

                    # Résultats
                    final_meta = new_store.get_metadata()
                    reindex_status.update(label="✅ Réindexation terminée avec succès !", state="complete")

                    st.success(
                        f"**Réindexation réussie !**\n\n"
                        f"- Modèle d'embedding : `{final_meta.get('embedding_model')}`\n"
                        f"- Documents indexés : **{final_meta.get('num_documents', 0)}**\n"
                        f"- Créé le : {(final_meta.get('created_at') or '')[:19]}"
                    )
                    st.session_state.show_reindex_form = False

                except Exception as exc:
                    reindex_status.update(label="❌ Erreur lors de la réindexation", state="error")
                    st.error(f"Erreur : {exc}")
                    logging.error("Erreur réindexation", exc_info=True)

                finally:
                    logging.getLogger().removeHandler(_log_handler)
                    if _log_handler.records:
                        with st.expander("📋 Logs détaillés", expanded=False):
                            st.code("\n".join(_log_handler.records), language="text")

    st.divider()

# Titre principal
st.title(f"📚 {APP_TITLE}")
st.caption(f"Posez vos questions sur {COMPANY_NAME}")

# Affichage de l'historique du chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Afficher les sources si elles existent pour les messages de l'assistant
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("Sources utilisées"):
                for i, source in enumerate(message["sources"]):
                    # Accès sécurisé aux métadonnées
                    meta = source.get("metadata", {})
                    st.markdown(f"**Source {i + 1}:** `{meta.get('source', 'N/A')}`")
                    st.markdown(f"*Score de similarité:* {source.get('score', 0.0):.2f}%")
                    if "raw_score" in source:
                        st.markdown(f"*Score brut:* {source.get('raw_score', 0.0):.4f}")
                    # st.markdown(f"*Catégorie:* `{meta.get('category', 'N/A')}`")
                    st.text_area(
                        f"Extrait {i + 1}",
                        value=source.get("text", "")[:500] + "...",
                        height=100,
                        disabled=True,
                        key=f"src_{message['timestamp']}_{i}",
                    )  # Clé unique pour éviter les conflits


# Zone de saisie utilisateur en bas
if prompt := st.chat_input("Posez votre question ici..."):
    # Obtenir la date du jour et le mois en cours
    now = datetime.datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_month = now.strftime("%B %Y")

    logging.info(f"La date courante est: {current_date} et le mois est: {current_month}")

    # Ajouter le message utilisateur à l'historique et l'afficher
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "timestamp": now.isoformat()}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Afficher un message d'attente
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 Recherche d'informations et génération de la réponse...")

        # --- Logique de traitement de la requête ---
        try:
            result = RAGPipeline(query_classifier, vector_store, client).run(
                question=prompt,
                k=num_docs,
                min_score=min_score,
                model=selected_model,
            )

            needs_rag = result.mode == "RAG"
            confidence = result.confidence
            reason = result.reason
            response_text = result.answer
            sources_for_log = result.sources

            mode_info = st.empty()
            if needs_rag and sources_for_log:
                mode_info.info(f"Mode RAG — {confidence:.2f} de confiance")
            elif not needs_rag:
                mode_info.info(f"Mode Direct — {confidence:.2f} de confiance")

            # 4. Afficher la réponse et les sources
            message_placeholder.markdown(response_text)

            # Afficher les sources si disponibles (mode RAG avec résultats)
            if sources_for_log:
                with st.expander("Sources utilisées"):
                    for i, source in enumerate(sources_for_log):
                        meta = source.get("metadata", {})
                        st.markdown(f"**Source {i + 1}:** `{meta.get('source', 'N/A')}`")
                        st.markdown(f"*Score de similarité:* {source.get('score', 0.0):.2f}%")
                        if "raw_score" in source:
                            st.markdown(f"*Score brut:* {source.get('raw_score', 0.0):.4f}")
                        # st.markdown(f"*Catégorie:* `{meta.get('category', 'N/A')}`")
                        st.text_area(
                            f"Extrait {i + 1}",
                            value=source.get("text", "")[:500] + "...",
                            height=100,
                            disabled=True,
                            key=f"src_new_{i}",
                        )  # Clé unique
            elif needs_rag:
                # Mode RAG sans résultats
                st.info(
                    "Aucune source pertinente n'a été trouvée dans la base de connaissances pour cette question."
                )
            else:
                # Mode Direct
                st.info(
                    "Réponse générée en mode direct, sans consultation de la base de connaissances."
                )

            # 5. Enregistrer l'interaction dans la base de données (sans feedback initial)
            # Ajouter des métadonnées sur le mode utilisé
            metadata = {
                "mode": "RAG" if needs_rag else "DIRECT",
                "confidence": confidence,
                "reason": reason,
            }

            interaction_id = log_interaction(
                query=prompt,
                response=response_text,
                sources=sources_for_log,  # Stocke la liste de dicts
                metadata=metadata,  # Ajouter les métadonnées sur le mode
            )
            st.session_state.last_interaction_id = interaction_id  # Garde l'ID pour le feedback
            logging.info(f"Interaction enregistrée avec ID: {interaction_id}")

            # Ajouter la réponse de l'assistant à l'historique pour affichage permanent
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources_for_log,  # Garder les sources pour réaffichage
                    "timestamp": now.isoformat(),
                    "interaction_id": interaction_id,  # Lier le message à l'ID BDD
                }
            )

        except Exception as e:
            # Vérifier si c'est une erreur API Mistral
            if hasattr(e, "status_code") and hasattr(e, "message"):
                logging.error(f"Erreur API Mistral: {e}")
                message_placeholder.error(
                    f"Une erreur s'est produite lors de la communication avec l'API Mistral: {e}"
                )
            else:
                logging.error(f"Erreur inattendue: {e}", exc_info=True)
                message_placeholder.error(f"Une erreur s'est produite: {e}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Erreur: {e}",
                    "sources": [],
                    "timestamp": now.isoformat(),
                    "interaction_id": None,
                }
            )
            st.session_state.last_interaction_id = None  # Pas d'ID si erreur avant log

# --- Section Feedback ---
# Placer le feedback après la boucle d'affichage et la zone de chat input
# On cible la *dernière* réponse de l'assistant pour le feedback
last_assistant_message = next(
    (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None
)

# Vérifie si la dernière réponse a un ID d'interaction associé
current_interaction_id = (
    last_assistant_message.get("interaction_id") if last_assistant_message else None
)

if current_interaction_id:
    # Utilisation de streamlit-feedback
    feedback = streamlit_feedback(
        feedback_type="thumbs",  # "thumbs" ou "faces"
        optional_text_label="[Optionnel] Commentaires :",
        key=f"feedback_{current_interaction_id}",  # Clé unique liée à l'interaction
        align="flex-start",  # Aligner à gauche
        on_submit=lambda x: logging.info(f"Feedback soumis: {x}"),  # Log pour débogage
    )

    # Traitement du feedback s'il est donné
    if feedback:
        # Convertir le feedback en valeur numérique et texte
        feedback_score = feedback.get("score")

        # Vérifier si le score est valide
        # Le composant streamlit_feedback peut renvoyer des emojis au lieu de "thumbs_up"/"thumbs_down"
        if feedback_score == "👍" or feedback_score == "thumbs_up":
            feedback_score = "positive"
        elif feedback_score == "👎" or feedback_score == "thumbs_down":
            feedback_score = "negative"
        else:
            logging.warning(f"Score de feedback invalide: {feedback_score}")
            feedback_score = None

        # 1 pour positif, 0 pour négatif
        feedback_value = (
            1 if feedback_score == "positive" else 0 if feedback_score == "negative" else None
        )

        # Texte pour la base de données ("positif" ou "négatif")
        feedback_text = (
            "positif"
            if feedback_score == "positive"
            else "négatif"
            if feedback_score == "negative"
            else "N/A"
        )

        # Emoji pour l'affichage dans l'interface
        feedback_emoji = (
            "👍"
            if feedback_score == "positive"
            else "👎"
            if feedback_score == "negative"
            else "N/A"
        )
        comment = feedback.get("text", None)

        # Mettre à jour l'interaction dans la base de données
        success = update_feedback(current_interaction_id, feedback_text, comment, feedback_value)
        if success:
            st.toast(f"Merci pour votre retour ({feedback_emoji}) !", icon="✅")
            # Optionnel: Désactiver les boutons après le premier clic pour éviter les soumissions multiples
            # Ceci est plus complexe à gérer avec la nature stateless de Streamlit sans callbacks avancés.
            # Pour la simplicité, on se contente de l'enregistrer. L'utilisateur peut re-cliquer mais seule la dernière valeur compte.
        else:
            st.toast("Erreur lors de l'enregistrement de votre retour.", icon="❌")

        # Optionnel : Effacer le feedback de l'état pour éviter re-soumission au re-run
        # st.session_state[f"feedback_{current_interaction_id}"] = None # Peut causer des pbs si mal géré

else:
    st.write("Posez une question pour pouvoir donner votre avis sur la réponse.")
