import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.io import arff 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

import os
os.environ["THREADPOOLCTL_VERBOSE"] = "0"


warnings.filterwarnings("ignore")


SPECIFIC_DATA_PATH = './dataset/dataset/artificial/' 

#Exécute k-means sur un jeu de données pour un nombre de clusters allant de k=2 à k=10 et trouve la meilleure solution selon une métrique
#Les métriques disponibles sont :
#    -coefficient de silhouette pour metrique=1
#    -coefficient de Calinski-Harabasz pour metrique=2
#    -coefficient de Davies-Bouldin pour metrique=3
#Des graphiques représentant la valeur des différentes métriques selon les solutions sont affichés
#Une représentation des clusters de la meilleure solution est affichée
def trouver_meilleur_kmeans_et_visualiser(file_name, metrique, max_k=10, data_path=SPECIFIC_DATA_PATH):

    
    
    
    print(f"--- Démarrage de l'optimisation k-Means pour {file_name} ---")
    
    
    full_path = os.path.join(data_path, file_name)

    
    try:
        if not os.path.exists(full_path):
             print(f"ERREUR: Fichier non trouvé à {full_path}")
             return
             
        
        databrut, _ = arff.loadarff(open(full_path, 'r')) 
        
        # On ignore le label de cluster .
        datanp = np.array([[x[0], x[1]] for x in databrut])
        
        print(f"Dataset chargé depuis {full_path}. {datanp.shape[0]} points trouvés.")
    except Exception as e:
        print(f"ERREUR lors du chargement de l'ARFF : {e}")
        return

    # Exécution de k-means de k=2 à k=10
    k_range = range(2, min(max_k + 1, datanp.shape[0]))
    results = [] 
    
    for k in k_range:
        # Entraînement du modèle K-Means 
        model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        model.fit(datanp)
        labels = model.labels_

        # Assure qu'il y a plus d'un cluster
        if len(np.unique(labels)) > 1: 
            score_sil = silhouette_score(datanp, labels)
            score_ch = calinski_harabasz_score(datanp, labels)
            score_db = davies_bouldin_score(datanp, labels)
            
            results.append({
                'k': k,
                'model': model,
                'Inertie': model.inertia_,
                'Silhouette': score_sil,
                'Calinski-Harabasz': score_ch,
                'Davies-Bouldin': score_db,
            })
            
    if not results:
        print("Aucun clustering valide trouvé.")
        return

    results_df = pd.DataFrame(results)
    
    # Sélection du meilleur k selon la métrique choisie
    if metrique == 1:
        best_solution = results_df.loc[results_df['Silhouette'].idxmax()]

    if metrique == 2:
        best_solution = results_df.loc[results_df['Calinski-Harabasz'].idxmax()]

    if metrique == 3:
        best_solution = results_df.loc[results_df['Davies-Bouldin'].idxmin()]



    
    best_k = best_solution['k']
    best_model = best_solution['model']
    
    print(f"\n--- MEILLEURE SOLUTION ---")
    print(f"Paramètre Optimal k: {int(best_k)}")

    print(f"Inertie pour ce k : {best_solution['Inertie']:.2f}")
    print(f"Score Silhouette : {best_solution['Silhouette']:.3f}")
    print(f"Score Calinski-Harabasz : {best_solution['Calinski-Harabasz']:.4f}")
    print(f"Score Davies-Bouldin : {best_solution['Davies-Bouldin']:.5f}")
    
    # Visualisation des 4 courbes des valeurs des métriques selon les valeurs de k

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Optimisation de k pour K-Means sur {file_name}", fontsize=16)

    
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
        ax.axvline(x=best_k, color='r', linestyle='--', linewidth=1, label=f'Optimal k={int(best_k)}')
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Visualisation du meilleur clustering
    
    plt.figure(figsize=(8, 8))
    plt.scatter(datanp[:, 0], datanp[:, 1], c=best_model.labels_, s=15, cmap='viridis')
    centroids = best_model.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=200, linewidths=3, color="red", label='Centroïdes')
    
    plt.title(f"Clustering Optimal K-Means : {file_name} (k={int(best_k)})")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend()
    plt.show()



trouver_meilleur_kmeans_et_visualiser("xclara.arff",3)
