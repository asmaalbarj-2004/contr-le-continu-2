
# 🛒 Compte Rendu : Analyse du Dataset Shopping Behaviour
<img src="ASSSSSSMAAAAAAA (1).jpg" style="height:200px;margin-right:150px"/>      
# AL BARJ ASMA 
---

## 1. Titre de l'étude

**Analyse du Comportement d'Achat des Clients – Dataset Shopping Behaviour**

---

## 2. Problématique

> **Quels sont les principaux facteurs qui influencent les décisions et habitudes d'achat des clients ?**

---

## 3. Objectif

Identifier et analyser les variables qui influencent le comportement d'achat afin de mieux comprendre les profils et décisions des clients.

---

## 4. Description des données

Le dataset **Shopping Behaviour** contient les informations de **3 900 clients**, incluant :

- **Caractéristiques personnelles** : ID client, âge, genre
- **Données transactionnelles** : produit acheté, catégorie, montant dépensé
- **Variables comportementales** : fréquence d'achat, avis, achats précédents
- **Facteurs contextuels** : couleur, saison, réduction, code promo, méthode de paiement, type de livraison

Ce dataset permet d'étudier **comment et pourquoi** les clients réalisent leurs achats.

---

## 5. Code Python utilisé

```python
# ======================================================
# 1️⃣ Importation des bibliothèques
# ======================================================
import kagglehub
from kagglehub import KaggleDatasetAdapter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import os

# ======================================================
# 2️⃣ Chargement du dataset via KaggleHub
# ======================================================
dataset_directory = "/kaggle/input/shopping-behaviour-dataset"
print(f"Files available in {dataset_directory}:")
for root, dirs, files in os.walk(dataset_directory):
    for file in files:
        print(os.path.join(root, file))

# IMPORTANT : mettre le nom exact du fichier CSV
file_path = "shopping_behavior_updated.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "grandmaster07/shopping-behaviour-dataset",
    file_path,
)

print("Aperçu du dataset:")
print(df.head())

# ======================================================
# 3️⃣ Vérification et Nettoyage des données
# ======================================================
print("\nValeurs manquantes :")
print(df.isnull().sum())

# Remplissage NA
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

df = df.drop_duplicates()

print("\nDataset après nettoyage :")
print(df.info())

# ======================================================
# 4️⃣ Statistiques descriptives
# ======================================================
print("\nStatistiques numériques :")
print(df.describe())

print("\nStatistiques catégorielles :")
print(df.describe(include="object"))

# ======================================================
# 5️⃣ Encodage des variables catégorielles
# ======================================================
label = LabelEncoder()

for col in df.select_dtypes(include="object"):
    df[col] = label.fit_transform(df[col])

# ======================================================
# 6️⃣ Matrice de corrélation
# ======================================================
plt.figure(figsize=(12, 9)) # Slightly increased figure size for better visibility
sns.heatmap(
    df.corr(),
    annot=True,     # Show the correlation values on the heatmap
    fmt=".2f",      # Format annotations to two decimal places
    cmap="coolwarm", # Colormap to visualize the correlation strength
    linewidths=.5   # Add lines between cells for better separation
)
plt.title("Matrice de corrélation des caractéristiques", fontsize=16) # More descriptive title
plt.show()


# ======================================================
# 7️⃣ Définition des variables & choix de la cible
# ======================================================
X = df.drop("Gender", axis=1)
y = df["Gender"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================================
# 8️⃣ Modélisation avec Random Forest
# ======================================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ======================================================
# 9️⃣ Évaluation du modèle
# ======================================================
y_pred = model.predict(X_test)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion brute :")
print(cm)

# 🔵 Matrice de confusion en graphique
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Homme", "Pred Femme"],
            yticklabels=["Réel Homme", "Réel Femme"])
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Valeurs réelles")
plt.show()

# ======================================================
# 🔟 Importance des variables + GRAPHE
# ======================================================
importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nImportance des variables :")
print(importances)

plt.figure(figsize=(10, 5))
sns.barplot(data=importances, x="Importance", y="Feature")
plt.title("Importance des variables (Random Forest)")
plt.show()
```

---

## 6. Analyse des étapes du code

### 6.1. Nettoyage des données
**📍 Référence : Étape 3️⃣ du code**

Le nettoyage a permis :

- ✅ Détecter et remplacer les valeurs manquantes
- ✅ Harmoniser les variables catégorielles
- ✅ Supprimer les doublons

**Résultat** : Une base fiable et prête pour les analyses statistiques et prédictives.

---

### 6.2. Statistiques descriptives
**📍 Référence : Étape 4️⃣ du code**

Les statistiques descriptives ont fourni une première vue d'ensemble :

- Répartition des âges
- Comportements d'achat
- Montants dépensés
- Tendances globales

**Objectif** : Identifier les profils types et comprendre les premières relations entre les variables.

---

### 6.3. Encodage des variables catégorielles
**📍 Référence : Étape 5️⃣ du code**

Les colonnes textuelles ont été transformées en valeurs numériques :

- **Exemples** : Saison, Méthode de paiement, Catégorie
- **Raison** : Permettre l'utilisation des modèles de machine learning

---

### 6.4. Matrice de corrélation
**📍 Référence : Étape 6️⃣ du code**

La matrice de corrélation met en évidence :

- 🔗 Les variables les plus liées entre elles
- ➕➖ Les relations positives ou négatives
- 🎯 Les facteurs pouvant avoir un impact sur la variable cible

**Importance** : Étape clé pour comprendre la structure du dataset.

---

### 6.5. Modélisation avec Random Forest
**📍 Référence : Étape 8️⃣ du code**

**Pourquoi Random Forest ?**

- ✅ Robustesse et performance
- ✅ Capacité à gérer des données mixtes (numériques + catégorielles)
- ✅ Pas de réglages complexes nécessaires
- ✅ Fournit l'importance des variables

**Résultat** : Permet de comprendre les facteurs influents dans les décisions d'achat.

---

### 6.6. Matrice de confusion et évaluation
**📍 Référence : Étape 9️⃣ du code**

Les résultats du modèle sont évalués grâce à :

- 📊 **Rapport de classification**
- 🔲 **Matrice de confusion**

**Métriques mesurées** :

- Précision (Precision)
- Rappel (Recall)
- Erreurs de prédiction

**Objectif** : Apprécier la qualité du modèle et son aptitude à comprendre le comportement client.

---

### 6.7. Importance des variables
**📍 Référence : Étape 🔟 du code**

Le modèle met en évidence les variables expliquant le mieux le comportement étudié.

**Résultat** : Identification des facteurs les plus déterminants dans les décisions d'achat :

- 💰 Montant dépensé
- 📦 Catégorie de produit
- 🌸 Saison
- 🔄 Fréquence d'achat
- *... et autres variables clés*

---

## 7. Conclusion générale

Cette étude du dataset **Shopping Behaviour** a permis d'explorer en détail les facteurs influençant les décisions d'achat des clients à travers une **analyse statistique, visuelle et prédictive**.


---

*Ce rapport constitue une analyse complète et structurée du comportement d'achat des clients, offrant des insights actionnables pour les équipes marketing et commerciales.*
