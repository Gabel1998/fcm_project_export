# Lokal Setup Guide — Mac (Apple Silicon) + IntelliJ

## Forudsætninger
- Mac med M1/M2/M3 chip
- IntelliJ IDEA (Community eller Ultimate)
- Homebrew installeret (https://brew.sh)

## Trin 1: Installer Python via Miniconda

Anbefalet: Miniconda til at administrere Python-miljøer.

```bash
# Hent Miniconda til Apple Silicon
brew install --cask miniconda

# Initialiser conda for din shell
conda init zsh   # eller bash hvis du bruger det
# Genstart terminalen efter dette
```

## Trin 2: Opret projekt-miljø

```bash
# Opret nyt miljø med Python 3.11 (ikke 3.13, mange ML-pakker halter)
conda create -n fcm-yolo python=3.11
conda activate fcm-yolo

# Installer PyTorch med MPS (Apple Silicon GPU) support
pip install torch torchvision torchaudio

# Installer projektafhængigheder
pip install ultralytics pandas openpyxl pillow matplotlib numpy scikit-image scipy
```

## Trin 3: Verificér MPS virker

Lav en test-fil `test_mps.py`:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Lille tensor på MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(3, 3, device=device)
    print(f"Tensor på MPS: {x}")
    print("MPS virker!")
else:
    print("MPS virker IKKE - check installation")
```

Kør: `python test_mps.py`

Hvis "MPS virker!" — du er klar.

## Trin 4: Verificér YOLO med MPS

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # downloader automatisk første gang
# Test prediction
results = model.predict("https://ultralytics.com/images/bus.jpg", device="mps")
print("YOLO + MPS virker!")
```

## Trin 5: Setup IntelliJ

1. Åbn IntelliJ
2. **File → New → Project from Existing Sources** → vælg projekt-mappen
3. **File → Project Structure → SDK** → Add new Python SDK
4. **Vælg "Conda Environment" → Existing → Browse →** find dit `fcm-yolo` miljø
   - Typisk i `/Users/<dit-navn>/miniconda3/envs/fcm-yolo/bin/python`
5. **Marker mappen som "Sources Root"** (højreklik → Mark Directory as)

## Trin 6: Test pipeline lokalt

```bash
# Aktivér miljøet
conda activate fcm-yolo

# Kør første script (genererer syntetiske data)
cd scripts/02_yolo_pipeline/
python 01b_realistic_synthetic_data.py

# Hvis det virker - kør de andre i rækkefølge
python 02_convert_to_yolo_format.py
python 03_train_yolo.py    # tager 5-10 min på MPS
python 04_predict_and_evaluate_landmarks.py
```

## Vigtige tilpasninger til lokalt

### I `03_train_yolo.py` skal du tilføje device-parameter

Find linjen `results = model.train(...)` og tilføj:

```python
results = model.train(
    ...,
    device="mps",   # ← TILFØJ DENNE
)
```

### I `04_predict_and_evaluate_landmarks.py` samme

```python
results = model.predict(img_path, conf=CONFIDENCE_THRESHOLD, verbose=False, device="mps")
```

### MODEL_PATH i step 4

Når du kører lokalt er stien anderledes end i Colab. Brug:

```python
import glob
matches = glob.glob('runs/**/best.pt', recursive=True)
MODEL_PATH = matches[0] if matches else None
```

## Trin 7: Git setup

```bash
git init
echo "data/videometer_raw/" >> .gitignore
echo "data/videometer_processed/" >> .gitignore
echo "*.hips" >> .gitignore
echo "runs/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".idea/" >> .gitignore
echo ".venv/" >> .gitignore
echo "*.pt" >> .gitignore
git add .
git commit -m "Initial commit"
```

Opret privat GitHub-repo, pushe der.

## Trouble­shooting

### "torch ikke fundet" eller "ultralytics import fejler"
→ Du står i forkert miljø. Kør `conda activate fcm-yolo` igen.

### YOLO træning er langsommere end forventet
→ Tjek at du har `device="mps"`. Uden MPS kører den på CPU og er meget langsommere.

### "MPS not available"
→ Du kører måske x86 Python under Rosetta. Tjek med `python -c "import platform; print(platform.machine())"`. Det skal sige `arm64`.

### IntelliJ ser ikke pakkerne
→ **File → Invalidate Caches and Restart**.
