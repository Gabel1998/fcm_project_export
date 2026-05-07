# Colab-træning

Træning på Colab T4 GPU er ~10× hurtigere end lokal MPS, og er især nyttig
når man eksperimenterer med data-mix eller hyperparametre.

**Vigtigt:** Kun **syntetiske data** her. Rigtige Videometer-data må aldrig
op til Colab pga. NDA — det er præcis derfor vores syntetiske generator
findes.

## Tilgængelige notebooks

### `train_robust.ipynb`

Træner en model der er robust på tværs af alle demo-parametre via **domain
randomization**: hvert træningsbillede får tilfældig støj og frø-størrelse,
så modellen ser hele variations-rummet under træning.

**Sådan kører du:**

1. Åbn [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → Upload** → vælg `train_robust.ipynb`
3. **Runtime → Change runtime type → T4 GPU** (gratis)
4. **Runtime → Run all** (Cmd/Ctrl + F9)
5. Vent ~5-10 min — sidste celle downloader `best.pt`
6. På din Mac:
   ```bash
   mkdir -p models/v2_robust
   mv ~/Downloads/best.pt models/v2_robust/
   ./switch_model.sh v2_robust
   fcm-demo
   ```

**Forventede resultater:** Lidt lavere precision på "kalibreret" (~0.93 vs
0.99) men markant højere på "svært" (~0.80 vs 0.44). Modellen er i alt taget
mere brugbar i en demo hvor parametrene varieres.

## Versions-match

Notebook'en pinner `ultralytics==8.4.47` så den downloadede `best.pt` er
kompatibel med din lokale installation. Hvis du opgraderer Ultralytics
lokalt, opdater også versionsnummeret i første kode-celle.
