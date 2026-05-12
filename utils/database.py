# utils/database.py
import datetime
import logging
import os
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String, Text, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import DATABASE_DIR, DATABASE_URL

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Crée le dossier de la base de données s'il n'existe pas
os.makedirs(DATABASE_DIR, exist_ok=True)

# Crée l'engine SQLAlchemy pour la base de données SQLite
# `check_same_thread=False` est nécessaire pour SQLite avec Streamlit/multithreading
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}, echo=False
)  # echo=True pour voir les requêtes SQL


# Base de déclaration SQLAlchemy 2.0
class Base(DeclarativeBase):
    pass


# Définit le modèle ORM pour la table des interactions
class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.now(datetime.UTC)
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str | None] = mapped_column(Text)
    sources: Mapped[Any] = mapped_column(JSON, nullable=True)
    query_metadata: Mapped[Any] = mapped_column(JSON, nullable=True)
    feedback: Mapped[str | None] = mapped_column(String(20))
    feedback_value: Mapped[int | None] = mapped_column(Integer)
    feedback_comment: Mapped[str | None] = mapped_column(Text)


# Crée la table dans la base de données si elle n'existe pas déjà
try:
    Base.metadata.create_all(engine)
    logging.info("Table 'interactions' vérifiée/créée dans la base de données.")
except SQLAlchemyError as e:
    logging.error(f"Erreur lors de la création/vérification de la table: {e}")

# Crée une factory de session pour interagir avec la base de données
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Fonction utilitaire pour obtenir une session de base de données."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_interaction(
    query: str,
    response: str,
    sources: list,
    metadata: dict | None = None,
    feedback: str | None = None,
    feedback_comment: str | None = None,
):
    """Enregistre une interaction dans la base de données.

    Args:
        query: Question de l'utilisateur
        response: Réponse générée
        sources: Liste des sources utilisées
        metadata: Métadonnées (mode, confiance, etc.)
        feedback: Feedback utilisateur
        feedback_comment: Commentaire de feedback

    Returns:
        ID de l'interaction enregistrée
    """
    db_session = SessionLocal()
    try:
        interaction = Interaction(
            query=query,
            response=response,
            sources=sources,  # SQLAlchemy gère la sérialisation JSON
            query_metadata=metadata,  # Métadonnées (mode, confiance, etc.)
            feedback=feedback,
            feedback_comment=feedback_comment,
        )
        db_session.add(interaction)
        db_session.commit()

        # Journaliser avec des informations sur le mode utilisé
        mode_info = ""
        if metadata and "mode" in metadata:
            mode_info = f", Mode: {metadata['mode']}"

        logging.info(
            f"Interaction enregistrée (Query: '{query[:50]}...'{mode_info}, Feedback: {feedback})"
        )
        return interaction.id  # Retourne l'ID de l'interaction enregistrée
    except SQLAlchemyError as e:
        logging.error(f"Erreur lors de l'enregistrement de l'interaction: {e}")
        db_session.rollback()  # Annule les changements en cas d'erreur
        return None
    finally:
        db_session.close()  # Ferme toujours la session


def get_all_interactions(limit: int = 100):
    """Récupère les dernières interactions de la base de données."""
    db_session = SessionLocal()
    try:
        interactions = (
            db_session.query(Interaction)
            .order_by(Interaction.timestamp.desc())
            .limit(limit)
            .all()
        )
        logging.info(f"{len(interactions)} interactions récupérées.")
        # Convertit les objets Interaction en dictionnaires pour une manipulation plus facile (ex: Pandas)
        return [
            {
                "id": inter.id,
                "timestamp": inter.timestamp,
                "query": inter.query,
                "response": inter.response,
                "sources": inter.sources,  # Déjà une liste de dicts (ou None)
                "metadata": inter.query_metadata,  # Métadonnées (mode, confiance, etc.)
                "feedback": inter.feedback,
                "feedback_comment": inter.feedback_comment,
            }
            for inter in interactions
        ]
    except SQLAlchemyError as e:
        logging.error(f"Erreur lors de la récupération des interactions: {e}")
        return []
    finally:
        db_session.close()


def update_feedback(
    interaction_id: int,
    feedback: str,
    feedback_comment: str | None = None,
    feedback_value: int | None = None,
):
    """Met à jour le feedback pour une interaction spécifique.

    Args:
        interaction_id: ID de l'interaction à mettre à jour
        feedback: Texte du feedback (emoji)
        feedback_comment: Commentaire optionnel
        feedback_value: Valeur numérique (1 pour positif, 0 pour négatif)

    Returns:
        True si la mise à jour a réussi, False sinon
    """
    db_session = SessionLocal()
    try:
        interaction = (
            db_session.query(Interaction)
            .filter(Interaction.id == interaction_id)
            .first()
        )
        if interaction:
            # Mise à jour des valeurs
            interaction.feedback = feedback
            interaction.feedback_value = feedback_value
            interaction.feedback_comment = feedback_comment

            # Enregistrer les modifications
            db_session.commit()
            logging.info(f"Feedback mis à jour pour l'interaction ID {interaction_id}")
            return True
        else:
            logging.warning(
                f"Interaction ID {interaction_id} non trouvée pour la mise à jour du feedback."
            )
            return False
    except SQLAlchemyError as e:
        logging.error(
            f"Erreur lors de la mise à jour du feedback pour l'interaction {interaction_id}: {e}"
        )
        db_session.rollback()
        return False
    finally:
        db_session.close()


def reset_database():
    """Supprime toutes les interactions de la base de données (utilisé pour les tests)."""
    db_session = SessionLocal()
    try:
        num_deleted = db_session.query(Interaction).delete()
        db_session.commit()
        logging.info(
            f"Base de données réinitialisée, {num_deleted} interactions supprimées."
        )
    except SQLAlchemyError as e:
        logging.error(f"Erreur lors de la réinitialisation de la base de données: {e}")
        db_session.rollback()
    finally:
        db_session.close()
