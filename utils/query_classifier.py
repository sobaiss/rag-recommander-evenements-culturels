"""
Module de classification des requêtes pour déterminer si une question nécessite RAG
"""

import logging
import re

from mistralai.client import Mistral

from utils.config import CHAT_MODEL, COMPANY_NAME, MISTRAL_API_KEY


class QueryClassifier:
    """
    Classe pour classifier les requêtes et déterminer si elles nécessitent RAG
    """

    def __init__(self):
        """
        Initialise le classificateur de requêtes
        """
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None

        # Mots-clés qui suggèrent un besoin de RAG
        self.events_company_keywords = [
            COMPANY_NAME.lower(),
            "événement",
            "festival",
            "concert",
            "exposition",
            "spectacle",
            "atelier",
            "conférence",
            "activité",
            "loisir",
            "culturel",
            "musique",
            "films",
        ]

        # Questions générales qui ne nécessitent pas de RAG
        self.general_patterns = [
            r"^(bonjour|salut|hello|coucou|hey|bonsoir)[\s\.,!]*$",
            r"^(merci|thanks|thank you|je te remercie)[\s\.,!]*$",
            r"^(comment ça va|ça va|comment vas-tu|comment allez-vous)[\s\.,!?]*$",
            r"^(au revoir|bye|à bientôt|à plus tard|à la prochaine)[\s\.,!]*$",
            r"^(qui es[- ]tu|qu'es[- ]tu|que fais[- ]tu|comment fonctionnes[- ]tu|tu es quoi)[\s\?]*$",
            r"^(aide|help|sos|besoin d'aide)[\s\.,!?]*$"
        ]

    def needs_rag(self, query: str) -> tuple[bool, float, str]:
        """
        Détermine si une requête nécessite RAG

        Args:
            query: Requête de l'utilisateur

        Returns:
            Tuple (besoin_rag, confiance, raison)
        """
        # Convertir la requête en minuscules pour la comparaison
        query_lower = query.lower()

        # 1. Vérifier les patterns de questions générales (salutations, remerciements, etc.)
        for pattern in self.general_patterns:
            if re.match(pattern, query_lower):
                return False, 0.95, "Question générale ou salutation"

        # 2. Vérifier la présence de mots-clés
        events_keywords_found = [kw for kw in self.events_company_keywords if kw in query_lower]
        if events_keywords_found:
            keywords_str = ", ".join(events_keywords_found)
            return True, 0.9, f"Contient des mots-clés liés aux évnènements: {keywords_str}"

        # 3. Utiliser le LLM pour les cas ambigus
        if self.mistral_client:
            return self._classify_with_llm(query)

        # Par défaut, utiliser RAG pour les questions longues (plus de 5 mots)
        words = query.split()
        if len(words) > 5:
            return True, 0.6, "Question complexe (plus de 5 mots)"

        # Par défaut, ne pas utiliser RAG
        return False, 0.5, "Aucun critère spécifique détecté"

    def _classify_with_llm(self, query: str) -> tuple[bool, float, str]:
        """
        Utilise le LLM pour classifier la requête

        Args:
            query: Requête de l'utilisateur

        Returns:
            Tuple (besoin_rag, confiance, raison)
        """
        try:
            system_prompt = f"""Vous êtes un classificateur de requêtes pour un assistant virtuel de la société de {COMPANY_NAME}.
Votre tâche est de déterminer si une question nécessite une recherche dans une base de connaissances spécifique à la société.

Répondez UNIQUEMENT par "RAG" ou "DIRECT" suivi d'une brève explication:
- "RAG" si la question porte sur des informations spécifiques à {COMPANY_NAME} (services, produits, événements, etc.)
- "DIRECT" si c'est une question générale, une salutation, ou une question qui ne nécessite pas d'informations spécifiques à la société.

Exemples:
Question: "Bonjour, comment ça va?"
Réponse: DIRECT - Simple salutation

Question: "Quels sont les concert prévus dans la ville de Paris?"
Réponse: RAG - Demande d'informations spécifiques à des événements connectés à {COMPANY_NAME}

Question: "Le concert de Giovanni à Paris est-il gratuit?"
Réponse: RAG - Question spécifique sur un événement de {COMPANY_NAME}

Question: "Qellle est la différence entre concert et événement?"
Réponse: DIRECT - Question générale de connaissance
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]

            response = self.mistral_client.chat.complete(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.1,  # Température basse pour des réponses cohérentes
                max_tokens=50  # Réponse courte suffisante
            )

            result = response.choices[0].message.content.strip()
            logging.info(f"Classification LLM pour '{query}': {result}")

            # Analyser la réponse
            if result.startswith("RAG"):
                confidence = 0.85  # Confiance élevée dans la décision du LLM
                reason = result.replace("RAG - ", "").replace("RAG-", "").replace("RAG:", "").strip()
                return True, confidence, reason
            elif result.startswith("DIRECT"):
                confidence = 0.85
                reason = result.replace("DIRECT - ", "").replace("DIRECT-", "").replace("DIRECT:", "").strip()
                return False, confidence, reason
            else:
                # Réponse ambiguë, utiliser RAG par défaut
                return True, 0.6, "Classification ambiguë, utilisation de RAG par précaution"

        except Exception as e:
            logging.error(f"Erreur lors de la classification avec LLM: {e}")
            # En cas d'erreur, utiliser RAG par défaut
            return True, 0.5, f"Erreur de classification: {str(e)}"
