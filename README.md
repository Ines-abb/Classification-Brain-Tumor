# SY09 - Data Mining - Projet de groupe 

## Introduction
Dans le cadre de l'UV SY09 - Data Mining nous avons mené un travail d'analyse du jeu de données "Brain Tumor". Ce dernier provient de Kaggle [Brain Tumor](https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor). Les tumeurs cérébrales causent environ 189 000 décès par an dans le monde. Ainsi, nous avons choisi de travailler sur la thématique de détection d'une présence de tumeur au cerveau, cherchant à répondre à la question suivante : est-il possible de détecter une tumeur cérébrale grâce aux caractéristiques obtenues par imagerie médicale du patient ?

## Dataset
Récapitulatif des colonnes de notre dataset afin d'en comprendre les notions associées

| Nom                       | Description                                         |
|---------------------------|-----------------------------------------------------|
| Moyenne (Mean)            | Moyenne des intensités des pixels.                 |
| Variance (Standard Deviation) | Dispersion autour de la moyenne.                  |
| Écart-Type (Standard Deviation) | Racine carrée de la variance.                    |
| Asymétrie (Skewness)      | Asymétrie autour de la moyenne.                    |
| Aplatissement (Kurtosis)  | "Pointicité" de la distribution des pixels.        |
| Contraste (Contrast)      | Variations texture et couleur.                     |
| Énergie (Energy)          | Mesure l'uniformité de l'image.                    |
| ASM (Angular Second Moment) | Uniformité des pixels.                           |
| Entropie (Entropy)        | Complexité, désordre.                              |
| Homogénéité (Homogeneity) | Proximité des valeurs.                             |
| Dissimilarité (Dissimilarity) | Différences pixels voisins.                      |
| Corrélation (Correlation) | Corrélation des pixels.                            |
| Rugosité (Coarseness)     | Rugosité de la texture.                            |
| Class                     | 1 = Tumor, 0 = Non-Tumor                           |

## Objectifs
- Explorer et analyser le dataset pour en comprendre la structure et les caractéristiques.
- Effectuer un prétraitement des données pour nettoyer, transformer et préparer les données pour la modélisation.
- Utiliser des méthodes supervisées et non supervisées telles que l'Analyse en Composantes Principales (ACP), K-means, les k plus proches voisins et la régression logistique pour la modélisation et l'analyse des données.


## Technologies Utilisées
- Python
- Jupyter Notebook
- Bibliothèques Python : Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, etc.

## Structure du Projet
Le projet est organisé comme suit :
- **Analyse** : Le notebook permet d'appliquer des méthodes d'analyse de datascience sur notre dataset
- **Rapport** : Un rapport résumant les résultats, les conclusions et les insights tirés du projet (rédigé en LaTeX).


## Contributeurs
- Tuân Le Minh
- Justine Pouget
- Ines Abbache
