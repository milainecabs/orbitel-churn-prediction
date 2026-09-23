# Orbitel – Prédiction du Churn  
Régression Logistique Optimisée & Décision Métier

Ce projet a été réalisé pour Orbitel, un opérateur télécom confronté à une hausse du churn.  
L’objectif est simple : **prédire quels clients risquent de partir** et **réduire le coût des erreurs de décision**.

---

## 🎯 Contexte & Enjeu Métier

Orbitel constate une augmentation du churn et souhaite :

- cibler les clients réellement à risque,
- éviter les actions marketing inutiles,
- réduire les pertes liées aux départs non détectés.

La variable cible est :

- **1 = client qui va churner**
- **0 = client fidèle**

### Pourquoi le coût métier est essentiel ?

Chaque erreur a un impact financier :

- **Faux Négatif (FN)** : le modèle dit “fidèle” alors que le client part → **200€ perdus**  
- **Faux Positif (FP)** : le modèle dit “churn” alors que le client reste → **50€ dépensés inutilement**

👉 L’objectif n’est pas d’avoir “le meilleur score”, mais **le coût le plus bas**.

---

## 📂 Données utilisées

Trois fichiers fournis :

- `churn_train.csv` — entraînement (avec la cible)
- `churn_test.csv` — validation (avec la cible)
- `churn_new.csv` — nouveaux clients (sans cible)

---

## 🧪 Partie 1 — Modèle initial Orbitel

Un modèle de base (régression logistique) est fourni.

### Résultats du modèle initial

- Recall churn : **17%** → 83% des churners non détectés  
- Problème de convergence :  
  > « STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT »  
- Variable `Segment` ignorée  
- Coût métier : **2350€**

### Conclusion

Le modèle initial **n’est pas exploitable** : il rate la majorité des clients à risque et génère un coût élevé.

---

## ⚙️ Partie 2 — Améliorations apportées

Pour corriger les limites du modèle initial, plusieurs améliorations ont été appliquées :

### 🔧 Préprocessing
- **StandardScaler** : garantit la convergence du modèle  
- **OneHotEncoding** : intègre la variable `Segment`  
- **max_iter augmenté** : supprime le message de non‑convergence

### 🔧 Modèle & optimisation
- **class_weight ajusté** : reflète l’importance des FN vs FP  
- **Pipeline sklearn** : même traitement sur train et test  
- **GridSearchCV + scorer métier** : optimisation directe du coût  
- **StratifiedKFold** : assure la présence de churners dans chaque fold

---

## 🧩 Partie 3 — Optimisation métier

Un expert métier indique qu’un client est **frustré** si :

- ses réclamations > 5  
- sa satisfaction < 2.5  

Cette variable a été ajoutée au modèle.

### 🔥 Optimisation du seuil de décision

Le seuil par défaut (0.5) n’est pas forcément optimal.  
Nous avons testé tous les seuils de **0.1 à 0.9**.

### Résultat

- Seuil optimal : **0.5**
- Coût métier : **1200€**
- FN = 5 | FP = 4

👉 **Économie : −1150€** par rapport au modèle initial.

### Justification (version manager)

Rater un client qui part coûte **4 fois plus cher** qu’agir inutilement.  
Le seuil optimal permet de :

- détecter davantage de churners,
- limiter les actions inutiles,
- réduire le coût global sans surcharger les équipes terrain.

---

## 📊 Bonus — Modèle alternatif : Gradient Boosting

Un second modèle a été testé pour comparaison.

### Résultats

- Seuil optimal : **0.2**
- Coût métier : **1300€**
- FN = 6 | FP = 2

Ce modèle offre un compromis différent, avec moins d’alertes inutiles.

---

## 📄 Partie 4 — Prédictions sur `churn_new.csv`

Le modèle final ajoute deux colonnes :

- `Churn_Proba` → probabilité de churn  
- `Churn_Pred` → 1 si churn, 0 sinon  

Résultat :  
**14 / 40 clients prédits churners**  
Taux de churn prédit : **35%**

---

## 🛠️ Stack technique

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  
- Pipeline sklearn  
- GridSearchCV  
- Google Colab

---

## 🚀 Exécution du projet

### 🔹 Option 1 — Google Colab

Le notebook utilise Google Drive :

```python
from google.colab import drive
drive.mount('/content/drive')

base_path = "/content/drive/MyDrive/Projet_ML/"
