# Konwersja głosu RVC

Praca inżynierska. Klaudiusz Waśko, Informatyka stosowana, Politechnika Krakowska.

Polski interfejs w Google Colab: przygotowanie nagrań, trening modelu głosu i zamiana głosu. Silnik to Applio 3.6.5 (MIT).

## Google Colab

1. Otwórz `notebooks/colab_app.ipynb` w Colabie.
2. `Runtime` → `Change runtime type` → **GPU**.
3. Uruchom komórki po kolei. Na końcu pojawi się link do interfejsu.

## Lokalnie

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

Windows: `.venv\Scripts\Activate.ps1` zamiast `source`.

Trening i zamiana głosu wymagają Applio w osobnym środowisku: `SETUP_APPLIO.md`.
