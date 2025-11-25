import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

def traiter_fichier(nom_fichier):
    path = './dataset/dataset/artificial/' + nom_fichier
    print("Traitement de : " + nom_fichier)
    
    data, meta = arff.loadarff(open(path, 'r'))
    
    liste_points = []
    for ligne in data:
        liste_points.append([ligne[0], ligne[1]])
    
    X = np.array(liste_points)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Recherche des meilleurs paramètres (Grid Search simple)
    meilleur_score = -1
    meilleurs_params = (0, 0)
    
    taille_range = range(2, 16)
    matrice_scores = np.zeros((len(taille_range), len(taille_range)))
    
    print("Recherche des paramètres en cours...")
    
    for i in range(len(taille_range)):
        for j in range(len(taille_range)):
            min_size = taille_range[i]
            min_samples = taille_range[j]
            
            # Création du modèle
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=min_samples)
            labels = clusterer.fit_predict(X_scaled)
            
            # On calcule le silhouette score seulement si on a trouvé des clusters
            # Si tout est en bruit (-1) ou un seul cluster, on met un mauvais score
            n_labels = len(set(labels))
            if -1 in labels:
                n_labels -= 1
            
            if n_labels > 1:
                score = silhouette_score(X_scaled, labels)
            else:
                score = -1
            
            matrice_scores[i, j] = score
            
            # On garde le meilleur
            if score > meilleur_score:
                meilleur_score = score
                meilleurs_params = (min_size, min_samples)

    best_size, best_samples = meilleurs_params
    print("Meilleurs paramètres trouvés : size=", best_size, " samples=", best_samples)
    print("Score :", round(meilleur_score, 3))

    # --- Graphique 1 : La Matrice ---
    plt.figure()
    plt.imshow(matrice_scores, origin='lower', cmap='viridis')
    plt.colorbar(label='Silhouette Score')
    plt.xlabel('Min Samples (index)')
    plt.ylabel('Min Cluster Size (index)')
    plt.title('Qualité du clustering selon les paramètres - ' + nom_fichier)
    plt.show()

    # --- Graphique 2 : Le résultat final ---
    model_final = hdbscan.HDBSCAN(min_cluster_size=best_size, min_samples=best_samples)
    labels_final = model_final.fit_predict(X_scaled)
    
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels_final, cmap='nipy_spectral', s=20)
    plt.title('Résultat HDBSCAN : ' + nom_fichier)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Liste des fichiers à tester
datasets = ["banana.arff", "spiralsquare.arff", "2d-20c-no0.arff"]

for d in datasets:
    traiter_fichier(d)