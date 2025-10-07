import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.io import arff 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

# On ignore les warnings pour une sortie console plus claire
warnings.filterwarnings("ignore")

# --- PARAMÈTRE DU CHEMIN D'ACCÈS SPÉCIFIQUE À TON PROJET ---
# Ce chemin correspond à : TP-APPRENTISSAGE-CL-REMBAUX-ELOUALI/dataset/dataset/artificial/
SPECIFIC_DATA_PATH = './dataset/dataset/artificial/' 

def trouver_meilleur_kmeans_et_visualiser(file_name, max_k=10, data_path=SPECIFIC_DATA_PATH):
    """
    Charge un dataset ARFF, exécute l'optimisation pour k-Means (k de 2 à max_k),
    et affiche les courbes de métriques ainsi que le clustering optimal.
    """
    
    print(f"--- Démarrage de l'optimisation k-Means pour {file_name} ---")
    
    # Construction du chemin complet
    full_path = os.path.join(data_path, file_name)

    # --- 1. LECTURE ET PRÉPARATION DES DONNÉES ARFF ---
    try:
        if not os.path.exists(full_path):
             print(f"ERREUR: Fichier non trouvé à {full_path}")
             return
             
        # Charge le fichier ARFF. On ignore les metadata (le second élément du tuple)
        databrut, _ = arff.loadarff(open(full_path, 'r')) 
        
        # Extraction des 2 features (X[0] et X[1]). On ignore le label de cluster (dernière colonne).
        datanp = np.array([[x[0], x[1]] for x in databrut])
        
        print(f"Dataset chargé depuis {full_path}. {datanp.shape[0]} points trouvés.")
    except Exception as e:
        print(f"ERREUR lors du chargement de l'ARFF : {e}")
        return

    # --- 2. BOUCLE D'ANALYSE POUR K = 2 à MAX_K ---
    k_range = range(2, min(max_k + 1, datanp.shape[0]))
    results = [] 
    
    for k in k_range:
        # Entraînement du modèle K-Means (10 initialisations pour la robustesse)
        model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        model.fit(datanp)
        labels = model.labels_
        
        if len(np.unique(labels)) > 1: # Assure qu'il y a plus d'un cluster pour les métriques
            score_sil = silhouette_score(datanp, labels)
            score_ch = calinski_harabasz_score(datanp, labels)
            score_db = davies_bouldin_score(datanp, labels)
            
            results.append({
                'k': k,
                'model': model,
                'Inertie': model.inertia_,
                'Silhouette': score_sil,
                'Calinski-Harabasz': score_ch,
                'Davies-Bouldin': score_db
            })
            
    if not results:
        print("Aucun clustering valide trouvé (Le dataset est peut-être trop petit).")
        return

    results_df = pd.DataFrame(results)
    
    # --- 3. SÉLECTION DU MEILLEUR K (Basé sur le Max Silhouette) ---
    best_solution = results_df.loc[results_df['Silhouette'].idxmax()]
    best_k = best_solution['k']
    best_model = best_solution['model']
    
    print(f"\n--- MEILLEURE SOLUTION (Critère: Max Silhouette) ---")
    print(f"Paramètre Optimal k: {int(best_k)}")
    print(f"Score Silhouette : {best_solution['Silhouette']:.3f}")
    print(f"Inertie pour ce k : {best_solution['Inertie']:.2f}")

    # --- 4. VISUALISATION DES 4 COURBES D'OPTIMISATION ---

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Optimisation de k pour K-Means sur {file_name}", fontsize=16)

    # Définition des métriques et leurs titres
    metrics_to_plot = [
        ('Inertie', 'Inertie (Méthode du Coude)'),
        ('Silhouette', 'Coefficient de Silhouette (Max est meilleur)'),
        ('Calinski-Harabasz', 'Calinski-Harabasz (Max est meilleur)'),
        ('Davies-Bouldin', 'Davies-Bouldin (Min est meilleur)')
    ]

    for i, (metric, title) in enumerate(metrics_to_plot):
        ax = axs[i // 2, i % 2]
        ax.plot(results_df['k'], results_df[metric], marker='o')
        ax.set_title(title)
        ax.set_xlabel('Nombre de clusters (k)')
        ax.set_ylabel(metric)
        # Ajout d'une ligne pour marquer le k optimal
        ax.axvline(x=best_k, color='r', linestyle='--', linewidth=1, label=f'Optimal k={int(best_k)}')
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- 5. VISUALISATION DU CLUSTERING FINAL OPTIMAL ---
    
    plt.figure(figsize=(8, 8))
    plt.scatter(datanp[:, 0], datanp[:, 1], c=best_model.labels_, s=15, cmap='viridis')
    centroids = best_model.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=200, linewidths=3, color="red", label='Centroïdes')
    
    plt.title(f"Clustering Optimal K-Means : {file_name} (k={int(best_k)})")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend()
    plt.show()


# --- LANCEMENT DE L'ANALYSE ---
# Lancement de l'analyse pour le premier dataset de succès (xclara.arff)
trouver_meilleur_kmeans_et_visualiser("jain.arff")