"""Shared fixtures: a tiny made-up corpus in three languages.

Nothing here downloads anything or touches a database. The texts are invented
for the tests (they are not the real articles), short enough to reason about by
hand, and aligned by ID across English, Portuguese and French the way the real
regulations are.
"""

from __future__ import annotations

import pytest

from eurlex_rag.index.chunking import build_chunks
from eurlex_rag.types import Chunk, Unit

# (id, title, heading, {lang: text})
_CORPUS = [
    (
        "AIA:art:5",
        {"en": "Prohibited practices", "pt": "Práticas proibidas", "fr": "Pratiques interdites"},
        {"en": "Chapter II", "pt": "Capítulo II", "fr": "Chapitre II"},
        {
            "en": "Systems that manipulate people through subliminal techniques are banned.",
            "pt": "São proibidos os sistemas que manipulam pessoas com técnicas subliminares.",
            "fr": "Les systèmes qui manipulent les personnes par des techniques subliminales "
            "sont interdits.",
        },
    ),
    (
        "AIA:art:6",
        {
            "en": "Classification of high-risk systems",
            "pt": "Classificação de sistemas de risco elevado",
            "fr": "Classification des systèmes à haut risque",
        },
        {"en": "Chapter III", "pt": "Capítulo III", "fr": "Chapitre III"},
        {
            "en": "A system is high-risk when it is a safety component of a regulated product.",
            "pt": "Um sistema é de risco elevado quando é um componente de segurança de um "
            "produto regulado.",
            "fr": "Un système est à haut risque lorsqu'il est un composant de sécurité d'un "
            "produit réglementé.",
        },
    ),
    (
        "AIA:art:6:1",
        {"en": "", "pt": "", "fr": ""},
        {"en": "", "pt": "", "fr": ""},
        {
            "en": "The provider must assess the safety component before placing it on the market.",
            "pt": "O prestador deve avaliar o componente de segurança antes de o colocar no "
            "mercado.",
            "fr": "Le fournisseur doit évaluer le composant de sécurité avant sa mise sur le "
            "marché.",
        },
    ),
    (
        "AIA:art:6:2",
        {"en": "", "pt": "", "fr": ""},
        {"en": "", "pt": "", "fr": ""},
        {
            "en": "Systems listed in the annex on biometric identification are also high-risk.",
            "pt": "Os sistemas enumerados no anexo sobre identificação biométrica são também de "
            "risco elevado.",
            "fr": "Les systèmes énumérés à l'annexe sur l'identification biométrique sont "
            "également à haut risque.",
        },
    ),
    (
        "AIA:rec:47",
        {"en": "", "pt": "", "fr": ""},
        {"en": "", "pt": "", "fr": ""},
        {
            "en": "Whereas safety components of machinery deserve particular attention.",
            "pt": "Considerando que os componentes de segurança das máquinas merecem especial "
            "atenção.",
            "fr": "Considérant que les composants de sécurité des machines méritent une "
            "attention particulière.",
        },
    ),
    (
        "AIA:anx:III",
        {
            "en": "High-risk areas",
            "pt": "Domínios de risco elevado",
            "fr": "Domaines à haut risque",
        },
        {"en": "", "pt": "", "fr": ""},
        {
            "en": "Biometric identification, education, employment and credit scoring.",
            "pt": "Identificação biométrica, educação, emprego e pontuação de crédito.",
            "fr": "Identification biométrique, éducation, emploi et notation de crédit.",
        },
    ),
    (
        "GDPR:art:17",
        {"en": "Right to erasure", "pt": "Direito ao apagamento", "fr": "Droit à l'effacement"},
        {"en": "Chapter III", "pt": "Capítulo III", "fr": "Chapitre III"},
        {
            "en": "A person can ask the controller to delete personal data without undue delay.",
            "pt": "Uma pessoa pode pedir ao responsável que apague os dados pessoais sem demora "
            "injustificada.",
            "fr": "Une personne peut demander au responsable d'effacer les données personnelles "
            "sans retard injustifié.",
        },
    ),
    (
        "GDPR:art:33",
        {
            "en": "Notification of a breach",
            "pt": "Notificação de uma violação",
            "fr": "Notification d'une violation",
        },
        {"en": "Chapter IV", "pt": "Capítulo IV", "fr": "Chapitre IV"},
        {
            "en": "The controller must notify the authority of a data breach within 72 hours.",
            "pt": "O responsável deve notificar a autoridade de uma violação de dados no prazo de "
            "72 horas.",
            "fr": "Le responsable doit notifier l'autorité d'une violation de données dans un "
            "délai de 72 heures.",
        },
    ),
]

LANGS = ("en", "pt", "fr")


def make_units(langs: tuple[str, ...] = LANGS) -> list[Unit]:
    """The synthetic corpus as units, in the requested languages."""
    return [
        Unit(id=uid, lang=lang, text=text[lang], title=title[lang], heading=heading[lang])
        for uid, title, heading, text in _CORPUS
        for lang in langs
    ]


@pytest.fixture
def units() -> list[Unit]:
    """All units in English, Portuguese and French."""
    return make_units()


@pytest.fixture
def en_units() -> list[Unit]:
    """English units only: what gets indexed."""
    return make_units(("en",))


@pytest.fixture
def chunks(en_units: list[Unit]) -> list[Chunk]:
    """Article-level chunks of the English units."""
    return build_chunks(en_units, strategy="article")


@pytest.fixture
def gold_records() -> list[dict]:
    """A few benchmark questions over the synthetic corpus, in the gold-file shape."""
    return [
        {
            "qid": "q0001",
            "tier": "A",
            "split": "dev",
            "gold": ["GDPR:art:33"],
            "source_lang": "pt",
            "text": {
                "en": "How long does a controller have to report a data breach?",
                "pt": "Quanto tempo tem o responsável para comunicar uma violação de dados?",
                "fr": "De combien de temps le responsable dispose-t-il pour signaler une "
                "violation de données ?",
            },
            "verified_by": "",
            "notes": "",
        },
        {
            "qid": "q0002",
            "tier": "A",
            "split": "dev",
            "gold": ["GDPR:art:17"],
            "source_lang": "fr",
            "text": {
                "en": "Can I ask a company to delete my personal data?",
                "pt": "Posso pedir a uma empresa que apague os meus dados pessoais?",
                "fr": "Puis-je demander à une entreprise d'effacer mes données personnelles ?",
            },
            "verified_by": "",
            "notes": "",
        },
        {
            "qid": "q0003",
            "tier": "B",
            "split": "test",
            "gold": ["AIA:art:6", "AIA:anx:III"],
            "source_lang": "pt",
            "text": {
                "en": "Is a biometric identification system high-risk?",
                "pt": "Um sistema de identificação biométrica é de risco elevado?",
                "fr": "Un système d'identification biométrique est-il à haut risque ?",
            },
            "verified_by": "",
            "notes": "",
        },
        {
            "qid": "q0004",
            "tier": "C",
            "split": "test",
            "gold": [],
            "source_lang": "pt",
            "text": {
                "en": "What is the corporate tax rate for software companies?",
                "pt": "Qual é a taxa de imposto sobre as sociedades para empresas de software?",
                "fr": "Quel est le taux d'impôt sur les sociétés pour les éditeurs de logiciels ?",
            },
            "verified_by": "",
            "notes": "",
        },
    ]
