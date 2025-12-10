# 🛒 Compte Rendu : Analyse du Dataset Shopping Behaviour

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
# Insérer ici l'intégralité du code généré précédemment
# Le code complet incluant toutes les étapes d'analyse
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

### Points clés de l'analyse :

1. **Pipeline structuré** : Nettoyage → Étude descriptive → Corrélation → Modélisation
2. **Tendances significatives** dégagées grâce aux visualisations et statistiques
3. **Variables impactantes** identifiées via l'analyse d'importance
4. **Modèle Random Forest** performant avec validation rigoureuse

### Résultats obtenus :

Le modèle Random Forest, soutenu par une matrice de confusion et une analyse de l'importance des variables, a fourni une **compréhension claire et exploitable** du comportement des clients.

### Applications pratiques :

Ces résultats permettent d'orienter les **futures stratégies marketing et commerciales** :

- 🎯 Ciblage personnalisé des clients
- 💡 Optimisation des campagnes promotionnelles
- 📈 Amélioration de l'expérience d'achat
- 🔮 Prédiction des comportements futurs

---

## 📚 Références techniques

- **Dataset** : Shopping Behaviour (3 900 clients)
- **Algorithme principal** : Random Forest Classifier
- **Métriques d'évaluation** : Accuracy, Precision, Recall, F1-Score
- **Outils utilisés** : Python, Pandas, Scikit-learn, Matplotlib, Seaborn

---

**Date de l'analyse** : 2024  
**Auteur** : [Votre nom]  
**Contact** : [Votre email]

---

*Ce rapport constitue une analyse complète et structurée du comportement d'achat des clients, offrant des insights actionnables pour les équipes marketing et commerciales.*