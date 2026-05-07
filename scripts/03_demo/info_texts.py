"""Centrale danske forklaringer til alle tekniske termer i demoen.

Holdt i én fil så de er nemme at redigere uden at jagte dem rundt i app.py.
"""

TOOLTIPS = {
    # Parametre
    "antal_frø": (
        "Hvor mange frø generatoren forsøger at placere på pladen. "
        "Tætpakkede billeder er sværere at detektere."
    ),
    "billed_størrelse": (
        "Modellen er trænet på 256×256. Mindre billeder gør prediction lettere "
        "men giver færre frø."
    ),
    "støj_niveau": (
        "Tilfældige reflektans-variationer i pladen. "
        "Kalibreret = som model blev trænet på."
    ),
    "frø_størrelse": (
        "Modellen er trænet på 'normal' størrelse (12-18 px lange). "
        "Mindre/større frø udfordrer modellen."
    ),

    # Visualiseringer
    "ground_truth": (
        "Den kendte sandhed fra generatoren — hvor frøene faktisk er placeret. "
        "Vores facit."
    ),
    "prediction": (
        "Modellens output: bounding boxes omkring hvert frø. "
        "Centrum af hver box = predicted landmark."
    ),

    # Metrics
    "precision": (
        "Andelen af modellens forudsigelser der rammer rigtigt. "
        "1.0 = ingen falske detektioner."
    ),
    "recall": (
        "Andelen af rigtige frø som modellen faktisk fandt. "
        "1.0 = ingen missede frø."
    ),
    "f1": (
        "Harmonisk gennemsnit af Precision og Recall — "
        "én score der balancerer begge."
    ),
    "median_fejl": (
        "Typisk afstand fra predicted center til rigtigt center. "
        "Rapporteret i mm via Videometers skala (0.05 mm/pixel)."
    ),
}
