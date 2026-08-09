# ============================================================
# DOC AI Trainer – Scientist Knowledge Retriever
# ============================================================

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import TrainerConfig

logger = logging.getLogger(__name__)


class ScientistRetriever:
    """
    A lightweight retriever that selects the most relevant scientist profiles
    for a given text input. This is used to 'implant' the expertise of
    historical figures into the training data.

    In the future, this can be replaced with a full vector database (e.g., Milvus)
    and more advanced embedding models. For now, it uses TF‑IDF for simplicity.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            lowercase=True,
        )
        self.scientist_profiles: List[Dict[str, Any]] = []
        self.profile_texts: List[str] = []
        self.tfidf_matrix: Optional[np.ndarray] = None

        # Load the scientist profiles
        self._load_profiles()

        if self.profile_texts:
            # Fit the vectorizer on all profile texts
            self.tfidf_matrix = self.vectorizer.fit_transform(self.profile_texts)
            logger.info(f"Initialized ScientistRetriever with {len(self.profile_texts)} profiles.")

    def _load_profiles(self) -> None:
        """
        Load the scientist profiles from a bundled JSON file.
        This is a temporary static list; future versions will read from MinIO/Postgres.
        """
        # Inline list of scientists (extracted from the Master Prompt)
        # Each profile includes name, field, and a brief description.
        self.scientist_profiles = [
            {"name": "Thales of Miletus", "field": "Geometry", "description": "Early geometry, intercept theorem."},
            {"name": "Pythagoras", "field": "Number Theory", "description": "Pythagorean theorem, number philosophy."},
            {"name": "Euclid", "field": "Geometry", "description": "Euclidean geometry, Elements structural logic."},
            {"name": "Archimedes", "field": "Calculus", "description": "Calculus foundations, pi approximation, buoyancy."},
            {"name": "Isaac Newton", "field": "Physics", "description": "Laws of motion, universal gravitation, calculus."},
            {"name": "Albert Einstein", "field": "Physics", "description": "Special relativity, general relativity, photoelectric effect."},
            {"name": "Marie Curie", "field": "Chemistry", "description": "Radioactivity, polonium and radium isolation."},
            {"name": "Alan Turing", "field": "Computer Science", "description": "Turing machine, computability theory, AI foundations."},
            {"name": "Ada Lovelace", "field": "Computer Science", "description": "First computer programmer, Bernoulli number algorithm."},
            {"name": "John von Neumann", "field": "Mathematics", "description": "Game theory, von Neumann architecture, cellular automata."},
            {"name": "Richard Feynman", "field": "Physics", "description": "Quantum electrodynamics, Feynman diagrams, quantum computing."},
            {"name": "Nikola Tesla", "field": "Engineering", "description": "AC induction motor, polyphase system, wireless power."},
            {"name": "Leonardo da Vinci", "field": "Engineering", "description": "Aerodynamic flight, structural mechanics, military machines."},
            {"name": "James Clerk Maxwell", "field": "Physics", "description": "Maxwell's equations, classical electromagnetism."},
            {"name": "Niels Bohr", "field": "Physics", "description": "Bohr model of the atom, Copenhagen interpretation."},
            {"name": "Charles Darwin", "field": "Biology", "description": "Theory of evolution, natural selection."},
            {"name": "Gregor Mendel", "field": "Biology", "description": "Genetic inheritance, laws of segregation."},
            {"name": "Linus Pauling", "field": "Chemistry", "description": "Nature of the chemical bond, quantum chemistry."},
            {"name": "Dmitri Mendeleev", "field": "Chemistry", "description": "Periodic table of elements."},
            {"name": "Alan Turing", "field": "Computer Science", "description": "Turing machine, computability theory, AI foundations."},
            {"name": "Grace Hopper", "field": "Computer Science", "description": "Compiler design, COBOL, high-level language translation."},
        ]

        # Build a combined text for each profile: name + field + description
        self.profile_texts = [
            f"{p['name']} {p['field']} {p['description']}"
            for p in self.scientist_profiles
        ]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve the top‑k most relevant scientist profiles for a given query string.

        Args:
            query: The input text (e.g., a training example or user query).
            top_k: Number of profiles to return. Defaults to config.scientist.top_k.

        Returns:
            A list of scientist profile dictionaries, sorted by relevance.
        """
        if not self.config.scientist.use_scientist_retrieval:
            return []

        if top_k is None:
            top_k = self.config.scientist.top_k

        if not self.profile_texts or self.tfidf_matrix is None:
            logger.warning("Scientist profiles not loaded; returning empty list.")
            return []

        # Vectorize the query
        query_vec = self.vectorizer.transform([query])

        # Compute cosine similarity with all profiles
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top‑k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return the corresponding profiles with similarity scores
        results = []
        for idx in top_indices:
            profile = self.scientist_profiles[idx].copy()
            profile["similarity"] = float(similarities[idx])
            results.append(profile)

        logger.debug(f"Retrieved {len(results)} scientist profiles for query: {query[:50]}...")
        return results

    def augment_training_example(self, text: str) -> str:
        """
        Augment a training example by prepending the most relevant scientist profiles.

        Args:
            text: The original training text.

        Returns:
            The augmented text with scientist context prepended.
        """
        if not self.config.scientist.use_scientist_retrieval:
            return text

        profiles = self.retrieve(text)
        if not profiles:
            return text

        # Build a context prefix
        context_lines = []
        for p in profiles:
            context_lines.append(f"[{p['name']} – {p['field']}]: {p['description']}")

        augmented_text = "\n".join(context_lines) + "\n\n" + text
        return augmented_text