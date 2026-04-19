# 🎗️ Predikce rakoviny prsu — Breast Cancer Classification

> **Typ úlohy**: Binární klasifikace  
> **Algoritmy**: SVM (Support Vector Machine) + Neuronová síť (MLP)  
> **Dataset**: Breast Cancer Wisconsin (Prognostic) — poskytnuta vyučujícím 
> **Jazyk**: Python 3.x | scikit-learn | Keras / TensorFlow

---

## 📌 Popis obchodní úlohy (Business Problem)

Rakovina prsu je jedním z nejčastějších onkologických onemocnění u žen na světě.
Včasná a přesná diagnostika — rozlišení nezhoubného a zhoubného nádoru —
má přímý vliv na volbu léčby a přežití pacientky.

**Cíl projektu**: Vytvořit ML model, který na základě **vlastností buněčných jader**
(získaných z biopsie) automaticky předpoví, zda je nádor nezhoubní nebo zhoubní.

**Využití v praxi**:
- Podpora rozhodování lékaře při diagnóze
- Druhý názor při hraniční diagnóze
- Zrychlení procesu vyhodnocení výsledků biopsie

---

## 🗄️ Dataset

| Vlastnost | Hodnota |
|---|---|
| Zdroj | Upravená verze datasetu poskytnutá vyučujícím (původ: Breast Cancer Wisconsin — UCI ML Repository) |
| Soubor | `data/rakovina_prsou.csv` |
| Počet záznamů | ~198 pacientek |
| Počet příznaků | 23 vstupních (po čištění: 13) |
| Cílová proměnná | `Tumor` — M (maligní = 1), B (benigní = 0) |
 
> ⚠️ Dataset byl předem upraven vyučujícím (částečné předzpracování) a **není totožný** s originálním UCI souborem.

### Příznaky datasetu

Příznaky jsou odvozeny z digitalizovaného snímku FNA (aspirát z tenké jehly) a popisují vlastnosti buněčných jader:

| Příznak | Popis |
|---|---|
| Radius | Průměrná vzdálenost od středu k okrajům jádra |
| Texture | Směrodatná odchylka hodnot šedi |
| Perimeter | Obvod jádra |
| Area | Plocha jádra |
| Smoothness | Místní odchylka délek poloměrů |
| Compactness | Kompaktnost tvaru jádra |
| Concavity | Závažnost konkávních částí obrysu |
| Concave Points | Počet konkávních bodů obrysu |
| Symmetry | Symetrie jádra |
| Fractal Dimension | Fraktální dimenze okraje |
| *Variance sloupce* | Variabilita výše uvedených příznaků napříč buňkami vzorku |

---


## 🔬 ML řešení

### Proč SVM?

SVM (Support Vector Machine) je vhodný pro:
- Menší datasety s dobře definovanými příznaky ✅
- Binární klasifikaci ✅  
- Situace kde je důležitá **interpretovatelnost hranice rozhodování** ✅
- Robustnost vůči odlehlým hodnotám (outliers) ✅

### Proč neuronová síť (MLP)?

MLP (Multi-Layer Perceptron) umožňuje:
- Zachycení nelineárních vztahů mezi příznaky ✅
- Potenciálně vyšší přesnost při správném nastavení ✅
- Srovnání s klasickým přístupem ✅

---

## 🏗️ Architektura projektu

```
BREAST-CANCER-PROJECT/
│
├── data/
│   └── rakovina_prsou.csv          # Dataset (UCI Breast Cancer Wisconsin)
│
├── notebooks/
│   ├── rakovina_prsou.ipynb        # Hlavní notebook s analýzou a modely
│   ├── scaler.bin                  # Uložený StandardScaler
│   └── svm_model.sav               # Uložený natrénovaný SVM model
│
├── pipeline.py                     # Celý ML pipeline (EDA → model → metriky)
├── requirements.txt                # Závislosti projektu
├── README.md                       # Tento soubor
└── .gitignore
```

---

## 🔄 ML Pipeline

```
Načtení dat (CSV)
       ↓
EDA — popis, histogramy, box ploty
       ↓
Čištění — odstranění ID, Time sloupců
       ↓
Kódování — Tumor: M→1, B→0
       ↓
Výběr příznaků — korelační matice, odstranění multikolinearity
       ↓
Standardizace — StandardScaler (μ=0, σ=1)
       ↓
Rozdělení dat — Train 75% / Val 15% / Test 10%
       ↓
┌──────────────────┬────────────────────────┐
│   SVM model      │   Neuronová síť (MLP)  │
│  GridSearchCV    │   Keras Sequential     │
└──────────────────┴────────────────────────┘
       ↓                        ↓
  Metriky SVM            Metriky MLP
       ↓                        ↓
         Porovnání modelů
               ↓
         Inference — vlastní data
               ↓
         Uložení výsledků (results.csv)
```

---

## 📊 Výsledky modelů

| Model | Val Accuracy | Poznámka |
|---|---|---|
| SVM Linear (baseline) | 90.59% | Základní model |
| SVM + GridSearchCV | 92.94% | Optimalizace hyperparametrů |
| MLP (Neural Network) | *viz notebook* | Sekce 4 v notebooku |

> **Nejlepší hyperparametry SVM** (GridSearchCV, cv=5):  
> Prohledávaný prostor: `poly` kernel (C: 1,5,10,50 | degree: 1-4 | gamma: 0.1-5) a `rbf` kernel

---

## ⚙️ Instalace a spuštění

```bash
# 1. Klonování repozitáře
git clone <url-repozitare>
cd BREAST-CANCER-PROJECT

# 2. Vytvoření virtuálního prostředí
python -m venv .venv
.venv\Scripts\activate        # Windows
# nebo: source .venv/bin/activate  # Linux/Mac

# 3. Instalace závislostí
pip install -r requirements.txt

# 4. Spuštění celého pipeline
python pipeline.py

# 5. Nebo otevřít notebook
jupyter notebook notebooks/rakovina_prsou.ipynb
```

---

## 📦 Závislosti

Viz `requirements.txt`. Klíčové knihovny:
- `pandas`, `numpy` — zpracování dat
- `scikit-learn` — SVM, StandardScaler, GridSearchCV, metriky
- `matplotlib`, `seaborn` — vizualizace
- `tensorflow` / `keras` — neuronová síť
- `joblib`, `pickle` — ukládání modelů

---

## 👩‍💻 Autor

Školní projekt — předmět Strojové učení  
Autor: Vasilisa Pozdniakova  
Dataset: upraven a poskytnut vyučujícím (původní zdroj: Breast Cancer Wisconsin Prognostic, UCI ML Repository)