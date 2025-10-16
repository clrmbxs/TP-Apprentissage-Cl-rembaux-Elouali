import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import arff 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import pandas as pd
# Importation de la librairie externe HDBSCAN
import hdbscan  

warnings.filterwarnings("ignore")
SPECIFIC_DATA_PATH = './dataset/dataset/artificial/' 

def optimiser_hdbscan_par_grille(file_name, size_range, samples_range, data_path=SPECIFIC_DATA_PATH):
    """
    Effectue une recherche par grille sur min_cluster_size et min_samples, 
    calcule le score Silhouette pour chaque combinaison et visualise le résultat.
    """
    print(f"\n--- Optimisation HDBSCAN par Grille pour {file_name} ---")
    full_path = os.path.join(data_path, file_name)

    # --- 1. LECTURE DES DONNÉES ET STANDARDISATION ---
    try:
        databrut, _ = arff.loadarff(open(full_path, 'r')) 
        X = np.array([[x[0], x[1]] for x in databrut])
        X_scaled = StandardScaler().fit_transform(X)
    except Exception as e:
        print(f"ERREUR lors du chargement de l'ARFF : {e}")
        return

    # --- 2. RECHERCHE PAR GRILLE ---
    results = []
    
    for min_size in size_range:
        for min_samples in samples_range:
            # S'assurer que min_samples >= min_cluster_size (bonne pratique)
            if min_samples < min_size:
                continue 

            try:
                # Entraînement du modèle HDBSCAN
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_size,
                    min_samples=min_samples,
                    prediction_data=True # Nécessaire si on veut utiliser les métriques
                )
                clusterer.fit(X_scaled)
                labels = clusterer.labels_

                # Calcul des métriques (HDBSCAN peut générer beaucoup de bruit, étiquette -1)
                # On calcule le Silhouette Score uniquement si plus d'un cluster est trouvé
                if len(np.unique(labels)) > 1 and labels.max() != -1:
                    score_sil = silhouette_score(X_scaled, labels)
                else:
                    score_sil = -1 # Mauvais score si pas de cluster ou un seul cluster

                results.append({
                    'min_cluster_size': min_size,
                    'min_samples': min_samples,
                    'Silhouette': score_sil,
                    'model': clusterer
                })
            except Exception as e:
                # Ignorer les combinaisons qui échouent
                continue 

    if not results:
        print("Aucun clustering HDBSCAN valide trouvé pour la plage de paramètres spécifiée.")
        return

    # --- 3. ANALYSE ET SÉLECTION OPTIMALE ---
    results_df = pd.DataFrame(results)
    best_solution = results_df.loc[results_df['Silhouette'].idxmax()]
    best_size = int(best_solution['min_cluster_size'])
    best_samples = int(best_solution['min_samples'])
    best_score = best_solution['Silhouette']
    
    print(f"\n--- MEILLEURE SOLUTION HDBSCAN (Max Silhouette) ---")
    print(f"Hyperparamètre min_cluster_size: {best_size}")
    print(f"Hyperparamètre min_samples: {best_samples}")
    print(f"Score Silhouette : {best_score:.4f}")

    # 4. VISUALISATION DU CLUSTERING FINAL (Basée sur le meilleur modèle)
    best_model = hdbscan.HDBSCAN(min_cluster_size=best_size, min_samples=best_samples)
    best_model.fit(X_scaled)
    final_labels = best_model.labels_

    plt.figure(figsize=(10, 8))
    # Utilisation des données originales (X) pour la visualisation
    plt.scatter(X[:, 0], X[:, 1], c=final_labels, s=15, cmap='viridis')
    plt.title(f"Clustering Optimal HDBSCAN: {file_name} (Size={best_size}, Samples={best_samples})")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()


# --- LANCEMENT DE L'ANALYSE ---

# ➡️ Choix du dataset de difficulté DBSCAN : spiralsquare.arff
# ➡️ Plages de recherche : 
#    - min_cluster_size de 5 à 15 (par pas de 1)
#    - min_samples de 5 à 15 (par pas de 1)
size_range = range(5, 16, 1)
samples_range = range(5, 16, 1)
optimiser_hdbscan_par_grille("spiralsquare.arff", size_range, samples_range)