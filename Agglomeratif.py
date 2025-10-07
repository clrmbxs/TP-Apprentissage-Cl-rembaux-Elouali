import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.io import arff 
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

warnings.filterwarnings("ignore")
SPECIFIC_DATA_PATH = './dataset/dataset/artificial/' 

def trouver_meilleur_clustering_agglomeratif(file_name, max_k=10, data_path=SPECIFIC_DATA_PATH):
    """
    Teste différentes méthodes de linkage pour trouver la combinaison (Linkage, k) 
    optimale basée sur le Score Silhouette et affiche le résultat final.
    """
    
    print(f"\n--- Recherche du Meilleur Clustering Agglomératif pour {file_name} ---")
    full_path = os.path.join(data_path, file_name)

    # --- 1. LECTURE DES DONNÉES ---
    try:
        databrut, _ = arff.loadarff(open(full_path, 'r')) 
        datanp = np.array([[x[0], x[1]] for x in databrut])
    except Exception as e:
        print(f"ERREUR lors du chargement de l'ARFF : {e}")
        return

    # --- 2. BOUCLE D'ANALYSE (Linkage et k) ---
    linkage_methods = ['ward', 'average', 'complete', 'single']
    all_results = []
    k_range = range(2, min(max_k + 1, datanp.shape[0]))
    
    for method in linkage_methods:
        for k in k_range:
            try:
                # La métrique 'euclidean' est utilisée avec tous les linkages pour la comparaison
                model = AgglomerativeClustering(n_clusters=k, linkage=method, metric='euclidean')
                model.fit(datanp)
                labels = model.labels_
                
                if len(np.unique(labels)) > 1:
                    score_sil = silhouette_score(datanp, labels)
                    score_ch = calinski_harabasz_score(datanp, labels)
                    score_db = davies_bouldin_score(datanp, labels)
                    
                    all_results.append({
                        'k': k,
                        'Linkage': method,
                        'model': model,
                        'Silhouette': score_sil,
                        'Calinski-Harabasz': score_ch,
                        'Davies-Bouldin': score_db
                    })
            except Exception:
                # Ignorer les erreurs spécifiques à Ward sur de très petites matrices (rare ici)
                continue
    
    if not all_results:
        print("Aucun clustering valide trouvé.")
        return

    results_df = pd.DataFrame(all_results)
    
    # --- 3. SÉLECTION DU MEILLEUR MODÈLE GLOBAL ---
    
    # Trouver le meilleur modèle basé sur le Score Silhouette
    best_solution = results_df.loc[results_df['Silhouette'].idxmax()]
    best_k = int(best_solution['k'])
    best_method = best_solution['Linkage']
    
    # Rerun du modèle optimal pour obtenir les labels exacts
    final_model = AgglomerativeClustering(n_clusters=best_k, linkage=best_method)
    final_model.fit(datanp)
    final_labels = final_model.labels_

    # --- 4. AFFICHAGE DES HYPERPARAMÈTRES ET SCORES FINAUX (pour la Section 1 du rapport) ---
    print("\n--- MEILLEURE SOLUTION GLOBALE AGGLOMÉRATIVE (Section 1 du Rapport) ---")
    print(f"Dataset : {file_name}")
    print("\nHypereparamètres:")
    print(f"  - Ascendant ou Descendant : Ascendant (Agglomératif par définition)")
    print(f"  - Expression de la distance : {best_method.capitalize()} Linkage (Minimisation de la [distance/variance])")
    print(f"  - Seuil de distance : Non utilisé (Coupe basée sur n_clusters={best_k})")
    
    print("\nSCORES DE QUALITÉ FINALE (pour justification):")
    print(f"  - Nombre de clusters (k): {best_k}")
    print(f"  - Score Silhouette : {best_solution['Silhouette']:.4f} (Max)")
    print(f"  - Score Calinski-Harabasz: {best_solution['Calinski-Harabasz']:.2f}")
    print(f"  - Score Davies-Bouldin: {best_solution['Davies-Bouldin']:.4f} (Min)")
    
    # Affichage d'une table complète pour comparaison des metrics
    print("\nTop 5 des résultats par Linkage et k (Score Silhouette):")
    print(results_df.sort_values(by='Silhouette', ascending=False).head(5)[['k', 'Linkage', 'Silhouette', 'Davies-Bouldin']].to_markdown(index=False, floatfmt=".4f"))


    # --- 5. VISUALISATION DU CLUSTERING FINAL OPTIMAL (Même style que K-Means) ---
    
    plt.figure(figsize=(8, 8))
    # Points de données colorés par cluster (comme K-Means)
    plt.scatter(datanp[:, 0], datanp[:, 1], c=final_labels, s=15, cmap='viridis')
    
    # NOTE: Nous n'affichons PAS les centroïdes car le clustering agglomératif n'en a pas.
    # L'affichage est donc visuellement le même que le résultat final de K-Means.
    
    plt.title(f"Clustering Agglomératif Optimal : {file_name} (Linkage={best_method}, k={best_k})")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()

# --- LANCEMENT DE L'ANALYSE ---

# ➡️ Lancement pour xclara.arff 
trouver_meilleur_clustering_agglomeratif("impossible.arff")