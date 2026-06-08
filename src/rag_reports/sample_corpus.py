"""Corpus SINTETICO di referti radiologici (fonte di verità versionata).

⚠️ ATTENZIONE: questi referti sono **generati artificialmente** a scopo dimostrativo.
NON sono dati reali di pazienti, non provengono da cartelle cliniche e non vanno usati
per scopi clinici. Ogni record è marcato con ``"synthetic": True``.

La pipeline può anche caricare un corpus reale e pubblico (es. Open-i / Indiana
University Chest X-ray reports, https://openi.nlm.nih.gov/) tramite
:func:`rag_reports.corpus.load_jsonl_corpus`.

Ogni referto ha la struttura:
    {
        "id": str,            # identificatore univoco (usato nelle citazioni [doc:ID])
        "modality": str,      # modalità di imaging
        "exam": str,          # tipo di esame
        "text": str,          # corpo del referto (Reperti + Conclusioni)
        "synthetic": True,    # SEMPRE True per questo corpus
    }
"""

from __future__ import annotations

# Referti sintetici. Linguaggio realistico ma interamente inventato.
SAMPLE_REPORTS: list[dict] = [
    {
        "id": "RX-CHEST-001",
        "modality": "Radiografia",
        "exam": "Torace PA ed LL",
        "text": (
            "Reperti: addensamento parenchimale a livello del lobo inferiore destro "
            "con broncogramma aereo, compatibile con processo di natura flogistica. "
            "Non versamento pleurico. Ili vascolari nei limiti. Profilo cardiaco "
            "nei limiti. Conclusioni: quadro radiografico suggestivo di polmonite "
            "del lobo inferiore destro."
        ),
        "synthetic": True,
    },
    {
        "id": "RX-CHEST-002",
        "modality": "Radiografia",
        "exam": "Torace PA",
        "text": (
            "Reperti: opacamento dell'emitorace sinistro con obliterazione del seno "
            "costofrenico omolaterale, riferibile a versamento pleurico di grado "
            "moderato. Parziale atelettasia del lobo inferiore sinistro. "
            "Conclusioni: versamento pleurico sinistro di entità moderata."
        ),
        "synthetic": True,
    },
    {
        "id": "RX-CHEST-003",
        "modality": "Radiografia",
        "exam": "Torace PA ed LL",
        "text": (
            "Reperti: campi polmonari liberi da addensamenti pleuro-parenchimali in "
            "fase attiva. Non versamento pleurico. Ombra cardiaca nei limiti per "
            "dimensioni e morfologia. Strutture ossee indenni. Conclusioni: esame "
            "del torace nei limiti della norma."
        ),
        "synthetic": True,
    },
    {
        "id": "RX-CHEST-004",
        "modality": "Radiografia",
        "exam": "Torace PA",
        "text": (
            "Reperti: marcato aumento dell'indice cardio-toracico con cardiomegalia. "
            "Ridistribuzione del circolo verso gli apici e sfumati addensamenti "
            "perilari ad ali di farfalla, compatibili con edema polmonare. "
            "Conclusioni: cardiomegalia con segni di scompenso cardiaco congestizio."
        ),
        "synthetic": True,
    },
    {
        "id": "RX-CHEST-005",
        "modality": "Radiografia",
        "exam": "Torace PA ed LL",
        "text": (
            "Reperti: pneumotorace apicale destro con falda aerea di circa 2 cm e "
            "parziale collasso del parenchima omolaterale. Non deviazione "
            "mediastinica. Conclusioni: pneumotorace destro di modesta entità, si "
            "consiglia controllo radiografico ravvicinato."
        ),
        "synthetic": True,
    },
    {
        "id": "RX-CHEST-006",
        "modality": "Radiografia",
        "exam": "Torace PA",
        "text": (
            "Reperti: nodulo polmonare solitario di circa 9 mm a margini regolari "
            "nel lobo superiore sinistro. Assenza di linfoadenopatie ilo-mediastiniche "
            "valutabili. Conclusioni: nodulo polmonare solitario, si raccomanda "
            "approfondimento con TC del torace."
        ),
        "synthetic": True,
    },
    {
        "id": "CT-CHEST-007",
        "modality": "Tomografia Computerizzata",
        "exam": "TC torace con mezzo di contrasto",
        "text": (
            "Reperti: difetto di riempimento endoluminale a carico delle arterie "
            "lobari inferiori bilaterali, riferibile a tromboembolia polmonare. "
            "Modesto versamento pleurico destro. Conclusioni: embolia polmonare "
            "bilaterale dei rami lobari inferiori."
        ),
        "synthetic": True,
    },
    {
        "id": "CT-CHEST-008",
        "modality": "Tomografia Computerizzata",
        "exam": "TC torace ad alta risoluzione",
        "text": (
            "Reperti: aree di ground-glass a distribuzione periferica e subpleurica "
            "bilaterale, con iniziali tralci fibrotici e bronchiectasie da trazione "
            "alle basi. Conclusioni: quadro compatibile con polmonite interstiziale "
            "in fase iniziale, da correlare con il dato clinico."
        ),
        "synthetic": True,
    },
    {
        "id": "CT-CHEST-009",
        "modality": "Tomografia Computerizzata",
        "exam": "TC torace senza contrasto",
        "text": (
            "Reperti: enfisema centrolobulare prevalente ai lobi superiori con bolle "
            "subpleuriche. Ispessimento delle pareti bronchiali compatibile con "
            "bronchite cronica. Conclusioni: enfisema polmonare in quadro di "
            "broncopneumopatia cronica ostruttiva."
        ),
        "synthetic": True,
    },
    {
        "id": "MR-BRAIN-010",
        "modality": "Risonanza Magnetica",
        "exam": "RM encefalo con e senza contrasto",
        "text": (
            "Reperti: area di alterato segnale iperintenso in DWI a livello della "
            "corona radiata sinistra con restrizione in ADC, compatibile con lesione "
            "ischemica acuta. Non emorragie intra- o extra-assiali. Conclusioni: "
            "ictus ischemico acuto della corona radiata sinistra."
        ),
        "synthetic": True,
    },
    {
        "id": "MR-BRAIN-011",
        "modality": "Risonanza Magnetica",
        "exam": "RM encefalo con contrasto",
        "text": (
            "Reperti: lesione espansiva intra-assiale temporale destra con "
            "enhancement disomogeneo ad anello ed edema perilesionale, con effetto "
            "massa sulle strutture adiacenti. Conclusioni: lesione espansiva "
            "cerebrale, in prima ipotesi neoplastica, da caratterizzare."
        ),
        "synthetic": True,
    },
    {
        "id": "US-ABD-012",
        "modality": "Ecografia",
        "exam": "Ecografia addome completo",
        "text": (
            "Reperti: colecisti distesa con multiple immagini iperecogene con cono "
            "d'ombra posteriore, riferibili a calcoli. Pareti colecistiche nei "
            "limiti. Vie biliari non dilatate. Conclusioni: colelitiasi multipla "
            "senza segni di colecistite acuta."
        ),
        "synthetic": True,
    },
    {
        "id": "US-ABD-013",
        "modality": "Ecografia",
        "exam": "Ecografia addome superiore",
        "text": (
            "Reperti: fegato di dimensioni aumentate con ecostruttura diffusamente "
            "iperecogena ('bright liver'), compatibile con steatosi epatica. Vena "
            "porta pervia. Conclusioni: steatosi epatica diffusa."
        ),
        "synthetic": True,
    },
    {
        "id": "RX-BONE-014",
        "modality": "Radiografia",
        "exam": "Radiografia del polso destro",
        "text": (
            "Reperti: rima di frattura a carico dell'epifisi distale del radio destro "
            "con minimo accavallamento dei monconi. Non interessamento della "
            "superficie articolare. Conclusioni: frattura distale del radio "
            "(frattura di Colles)."
        ),
        "synthetic": True,
    },
    {
        "id": "MR-SPINE-015",
        "modality": "Risonanza Magnetica",
        "exam": "RM rachide lombo-sacrale",
        "text": (
            "Reperti: ernia discale postero-laterale sinistra a livello L5-S1 con "
            "impronta sul sacco durale e sulla radice S1 omolaterale. Disidratazione "
            "discale multilivello. Conclusioni: ernia discale L5-S1 con conflitto "
            "radicolare S1 sinistro."
        ),
        "synthetic": True,
    },
]


def get_sample_reports() -> list[dict]:
    """Restituisce una copia profonda del corpus sintetico in-code."""
    return [dict(r) for r in SAMPLE_REPORTS]
