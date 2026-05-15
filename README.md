# CLIP from Scratch : Apprentissage Multimodal Contrastif

Ce dépôt contient une implémentation complète et détaillée de **CLIP** (*Contrastive Language-Image Pre-training*) développée avec **PyTorch**. L'objectif est d'aligner des représentations visuelles et textuelles dans un espace latent commun.

**Auteur :**  Landry NOUMBISSI  
**Domaine :** Master 2 Recherche | Science des Données & IA

---

## 🚀 Vue d'ensemble du Projet
L'implémentation repose sur le principe de l'apprentissage contrastif : maximiser la similarité cosinus entre les paires (image, texte) correspondantes tout en la minimisant pour les paires incorrectes au sein d'un même batch.

### Fonctionnalités Clés :
- **Architecture Modulaire :** Encodeurs de vision et de texte indépendants.
- **Projection Heads :** Alignement des dimensions des vecteurs d'embeddings.
- **Loss Symétrique :** Implémentation de la *Contrastive Loss* pour l'entraînement multimodal.
- **Intégration HuggingFace :** Utilisation des bibliothèques `transformers` et `datasets`.

---

## 🏗️ Architecture du Système

Le modèle se compose de trois briques fondamentales :

1. **Image Encoder :** Utilise généralement un transformateur de vision (ViT) ou un ResNet pour extraire les caractéristiques visuelles.
2. **Text Encoder :** Basé sur un modèle de langage (type DistilBERT ou BERT) pour transformer les descriptions textuelles en vecteurs denses.
3. **Projection Head :** Des couches linéaires qui projettent les sorties des deux encodeurs vers un espace de dimension commune (ex: 256 ou 512).

---

## 📦 Installation et Dépendances

Le projet nécessite les bibliothèques suivantes pour la gestion des données massives et le calcul tensoriel :

```bash
pip install torch torchvision transformers datasets pandas tqdm aiohttp
