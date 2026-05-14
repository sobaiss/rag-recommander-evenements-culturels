"""
Module de classification des requêtes pour déterminer si une question nécessite RAG
"""

import re


class QueryClassifier:
    """
    Classe pour classifier les requêtes et déterminer si elles nécessitent RAG
    """

    def __init__(self):
        """
        Initialise le classificateur de requêtes
        """

        # Mots-clés qui suggèrent un besoin de RAG
        self.events_company_keywords = [
            "événement",
            "evenement",
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
            r"^(aide|help|sos|besoin d'aide)[\s\.,!?]*$",
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
        events_keywords_found = [
            kw for kw in self.events_company_keywords if kw in query_lower
        ]
        if events_keywords_found:
            keywords_str = ", ".join(events_keywords_found)
            return (
                True,
                0.9,
                f"Contient des mots-clés liés aux évènements: {keywords_str}",
            )

        # Par défaut, utiliser RAG pour les questions longues (plus de 5 mots)
        words = query.split()
        if len(words) > 5:
            return True, 0.6, "Question complexe (plus de 5 mots)"

        # Par défaut, ne pas utiliser RAG
        return False, 0.5, "Aucun critère spécifique détecté"
