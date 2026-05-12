# FCM Composite Pipeline — selvkørende workflow
#
# Alle kommandoer kan kaldes med: make <target>
# Variabler kan overrides: make gen DATASET=v2_dense
#
# Typisk arbejdsgang (uden internet, kører over natten):
#   make gen DATASET=v2_dense          # generér nyt dataset (~30 min)
#   make train DATASET=v2_dense NAME=v5_dense    # træn YOLO (~1-3 timer)
#   make watch NAME=v5_dense           # se live om træning går godt
#   make eval NAME=v5_dense            # eval bagefter

VENV    := .venv/bin
PYTHON  := $(VENV)/python

# Default-værdier — overrides med fx: make gen DATASET=v2_dense
DATASET ?= composite_v1
NAME    ?= v4_composite
EPOCHS  ?= 80
IMGSZ   ?= 1280
BATCH   ?= 4

COMPOSITE_DIR := scripts/02_yolo_pipeline/composite_generator
DATA_DIR      := data/$(DATASET)
RUN_DIR       := runs/train/$(NAME)

.PHONY: help gen gen-small preview train watch eval predict demo status test clean-tmp

help:
	@echo "FCM Composite Pipeline — selvkørende kommandoer"
	@echo ""
	@echo "  make help                           Vis denne hjælp"
	@echo "  make status                         Vis hvilke datasæt/modeller findes"
	@echo ""
	@echo "  Generér data:"
	@echo "  make gen [DATASET=v1]               Generér fuldt 2000-composite dataset (default config)"
	@echo "  make gen-config [CONFIG=t_heavy] [DATASET=v2]"
	@echo "                                      Generér med variant-config fra configs/"
	@echo "                                      Varianter: t_heavy, dense_only, sparse"
	@echo "  make gen-small [DATASET=v1]         Generér 50 composites (quick check)"
	@echo "  make preview [DATASET=v1]           Åbn preview.png for et dataset"
	@echo ""
	@echo "  Træn:"
	@echo "  make train [DATASET=v1] [NAME=v4]   Træn YOLO på dataset"
	@echo "  make watch [NAME=v4]                Live monitorering under træning"
	@echo "  make eval [NAME=v4]                 Eval på val-set efter træning"
	@echo "  make predict [NAME=v4]              Predict på sample-billede"
	@echo ""
	@echo "  Versionér model:"
	@echo "  make save-model [NAME=v4] [MODEL_DIR=v4]"
	@echo "                                      Arkivér best.pt + plots + notes til models/<MODEL_DIR>/"
	@echo "  make switch-model MODEL_DIR=v4      Skift active-symlink til models/<MODEL_DIR>/"
	@echo ""
	@echo "  Demo & dev:"
	@echo "  make demo                           Start Streamlit-demo"
	@echo "  make test                           Kør pytest på composite-pipeline"
	@echo "  make clean-tmp                      Slet /tmp/composite_* sanity checks"
	@echo ""
	@echo "  Variabler (override på kommandolinjen):"
	@echo "    DATASET=$(DATASET)   NAME=$(NAME)   EPOCHS=$(EPOCHS)"
	@echo "    IMGSZ=$(IMGSZ)       BATCH=$(BATCH)"
	@echo ""
	@echo "  Eksempel: lav et nyt dataset med eget navn + træn på det:"
	@echo "    make gen DATASET=v2_t_heavy"
	@echo "    make train DATASET=v2_t_heavy NAME=v5_t_heavy"

status:
	@echo "=== Datasæt (data/composite_*) ==="
	@if ls -d data/composite_* >/dev/null 2>&1; then \
		for d in data/composite_*; do \
			n=$$(ls $$d/images/train 2>/dev/null | wc -l | tr -d ' '); \
			v=$$(ls $$d/images/val 2>/dev/null | wc -l | tr -d ' '); \
			echo "  $$d:  $$n train / $$v val"; \
		done; \
	else echo "  (ingen)"; fi
	@echo ""
	@echo "=== Trænings-runs (runs/train/*) ==="
	@if ls -d runs/train/* >/dev/null 2>&1; then \
		for r in runs/train/*; do \
			if [ -f $$r/results.csv ]; then \
				e=$$(tail -n +2 $$r/results.csv | wc -l | tr -d ' '); \
				echo "  $$r:  $$e epochs"; \
			else \
				echo "  $$r:  (ingen results endnu)"; \
			fi; \
		done; \
	else echo "  (ingen)"; fi
	@echo ""
	@echo "=== Modeller (models/v*) ==="
	@ls -d models/v* 2>/dev/null || echo "  (ingen)"
	@echo ""
	@echo "=== Aktiv model ==="
	@ls -la models/active 2>/dev/null | awk '{print "  ", $$NF}' || echo "  (ingen)"
	@echo ""
	@echo "=== Kører træning lige nu? ==="
	@MATCH=$$(ps aux | grep -E "03b_train_composite|03_train_yolo|ultralytics" | grep -v grep); \
	if [ -n "$$MATCH" ]; then \
		echo "  ↑ Ja"; \
		echo "$$MATCH" | awk '{printf "    PID %s, CPU %s%%, RAM %s%%\n", $$2, $$3, $$4}'; \
	else \
		echo "  Nej"; \
	fi

gen:
	@echo "Genererer fuldt dataset i $(DATA_DIR)/ (~30-40 min)"
	$(PYTHON) $(COMPOSITE_DIR)/generate_dataset.py \
		--output-dir $(DATA_DIR)

# Generér med eget config — vælg variant fra configs/-mappen.
# Tilgængelige varianter: t_heavy, dense_only, sparse
# Brug: make gen-config CONFIG=t_heavy DATASET=v2_t_heavy
CONFIG ?= t_heavy
gen-config:
	@test -f $(COMPOSITE_DIR)/configs/$(CONFIG).yaml || (echo "❌ $(COMPOSITE_DIR)/configs/$(CONFIG).yaml findes ikke. Varianter: t_heavy, dense_only, sparse"; exit 1)
	@echo "Genererer $(DATASET) med config $(CONFIG).yaml"
	$(PYTHON) $(COMPOSITE_DIR)/generate_dataset.py \
		--config $(COMPOSITE_DIR)/configs/$(CONFIG).yaml \
		--output-dir $(DATA_DIR)

gen-small:
	@echo "Genererer mini-dataset (50 composites) i $(DATA_DIR)_small/"
	$(PYTHON) $(COMPOSITE_DIR)/generate_dataset.py \
		--output-dir $(DATA_DIR)_small \
		--num-composites 50

preview:
	@test -f $(DATA_DIR)/preview.png || (echo "Ingen preview i $(DATA_DIR) — kør 'make gen' først"; exit 1)
	@open $(DATA_DIR)/preview.png

train:
	@test -f $(DATA_DIR)/data.yaml || (echo "❌ Mangler $(DATA_DIR)/data.yaml — kør 'make gen DATASET=$(DATASET)' først"; exit 1)
	@echo "Træner YOLOv11n på $(DATA_DIR) som $(NAME) (~1-3 timer)"
	$(PYTHON) scripts/02_yolo_pipeline/03b_train_composite.py \
		--dataset $(DATASET) --name $(NAME) \
		--epochs $(EPOCHS) --imgsz $(IMGSZ) --batch $(BATCH)

watch:
	./scripts/watch_training.sh $(NAME)

eval:
	@test -d $(RUN_DIR) || (echo "❌ Ingen træning fundet på $(RUN_DIR)"; exit 1)
	@echo "Evaluering på val-set for $(NAME)"
	$(PYTHON) -c "from ultralytics import YOLO; \
	import json; \
	model = YOLO('$(RUN_DIR)/weights/best.pt'); \
	r = model.val(data='$(DATA_DIR)/data.yaml', imgsz=$(IMGSZ), device='mps'); \
	print(json.dumps({'precision': float(r.box.mp), 'recall': float(r.box.mr), 'mAP50': float(r.box.map50), 'mAP50-95': float(r.box.map)}, indent=2))"

predict:
	@test -d $(RUN_DIR) || (echo "❌ Ingen træning fundet på $(RUN_DIR)"; exit 1)
	@SAMPLE=$$(ls $(DATA_DIR)/images/val/*.png 2>/dev/null | head -1); \
	if [ -z "$$SAMPLE" ]; then echo "❌ Ingen val-billede fundet"; exit 1; fi; \
	echo "Predicter på $$SAMPLE"; \
	$(PYTHON) -c "from ultralytics import YOLO; \
	YOLO('$(RUN_DIR)/weights/best.pt').predict('$$SAMPLE', save=True, imgsz=$(IMGSZ), device='mps')"
	@echo "Resultat i runs/detect/predict/ (åbn nyeste)"

# Arkiver en trænet model til models/<MODEL_DIR>/ med notes og plots
# Brug: make save-model NAME=v5_t_heavy MODEL_DIR=v5_t_heavy
# Default MODEL_DIR er det samme som NAME hvis ikke specificeret
MODEL_DIR ?= $(NAME)
save-model:
	@test -f $(RUN_DIR)/weights/best.pt || (echo "❌ best.pt ikke fundet på $(RUN_DIR)/weights/"; exit 1)
	@echo "Arkiverer $(NAME) → models/$(MODEL_DIR)/"
	@mkdir -p models/$(MODEL_DIR)
	@cp $(RUN_DIR)/weights/best.pt models/$(MODEL_DIR)/best.pt
	@cp $(RUN_DIR)/results.png models/$(MODEL_DIR)/results.png 2>/dev/null || echo "  (ingen results.png at kopiere)"
	@cp $(RUN_DIR)/results.csv models/$(MODEL_DIR)/results.csv 2>/dev/null || true
	@if [ ! -f models/$(MODEL_DIR)/notes.md ]; then \
		printf "# %s\n\n**Trænet:** $$(date +%%Y-%%m-%%d)\n**Dataset:** $(DATASET)\n**Arkitektur:** YOLOv11n\n\n## Trænings-config\n\n- imgsz: $(IMGSZ)\n- batch: $(BATCH)\n- epochs: $(EPOCHS)\n- device: mps\n\n## Metrics (val-set)\n\nKør \`make eval NAME=$(NAME)\` og indsæt tal her.\n\n## Noter\n\n_Tilføj observationer her: hvad gik godt, hvad gik dårligt, sammenligning med tidligere modeller._\n" "$(MODEL_DIR)" > models/$(MODEL_DIR)/notes.md; \
		echo "  Genererede notes.md template"; \
	else \
		echo "  notes.md findes — ikke overskrevet"; \
	fi
	@echo ""
	@echo "Færdig. For at skifte aktiv model til denne:"
	@echo "  ./switch_model.sh $(MODEL_DIR)"
	@echo ""
	@echo "Eller manuelt:"
	@echo "  cd models && ln -sfn $(MODEL_DIR) active && cd .."

switch-model:
	@./switch_model.sh $(MODEL_DIR)

demo:
	$(VENV)/streamlit run scripts/03_demo/app.py

test:
	$(VENV)/pytest scripts/02_yolo_pipeline/composite_generator/tests/ -v

clean-tmp:
	rm -rf /tmp/composite_*
	@echo "Slettede /tmp/composite_*"
