import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import arff 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

warnings.filterwarnings("ignore")
SPECIFIC_DATA_PATH = './dataset/dataset/artificial/' 

# --- PARAMÈTRES POUR FORCER L'ÉCHEC ---
# Le but est de montrer que DBSCAN ne peut pas capturer les spirales peu denses 
# sans fusionner les carrés. On met un Epsilon petit.
EPSILON_ECHEC = 0.05 
MIN_PTS = 5 # On garde MinPts à 5

def visualiser_dbscan_echec(file_name, eps, min_pts, data_path=SPECIFIC_DATA_PATH):
    """
    Exécute DBSCAN avec un Epsilon trop petit pour le dataset spiralsquare.arff
    afin de démontrer sa limite sur la densité variable.
    """
    
    print(f"\n--- VISUALISATION DBSCAN (ÉCHEC VOLONTAIRE) pour {file_name} ---")
    full_path = os.path.join(data_path, file_name)

    # --- 1. LECTURE DES DONNÉES ET STANDARDISATION ---
    try:
        databrut, _ = arff.loadarff(open(full_path, 'r')) 
        X = np.array([[x[0], x[1]] for x in databrut])
        X_scaled = StandardScaler().fit_transform(X)
    except Exception as e:
        print(f"ERREUR lors du chargement de l'ARFF : {e}")
        return

    # --- 2. ENTRAÎNEMENT DU MODÈLE (Avec paramètres d'échec) ---
    model = DBSCAN(eps=eps, min_samples=min_pts)
    model.fit(X_scaled)
    labels = model.labels_

    # --- 3. ANALYSE ET MÉTRIQUES FINALES ---
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # --- 4. AFFICHAGE DE L'ÉCHEC ---
    print("\nSTATUT DU CLUSTERING DBSCAN :")
    print(f"  - Epsilon utilisé: {eps:.4f} (Volontairement trop petit)")
    print(f"  - Clusters trouvés: {n_clusters}")
    print(f"  - Points de bruit (label -1): {n_noise} (Les spirales devraient être ici)")

    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=15, cmap='viridis')
    
    plt.title(f"DBSCAN ÉCHEC sur {file_name} (eps={eps:.3f}, MinPts={min_pts})")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()


visualiser_dbscan_echec("Zelnik2.arff", eps=EPSILON_ECHEC, min_pts=MIN_PTS) 

