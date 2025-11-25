import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.io import arff

datasets = [
    "banana.arff",
    "Zelnik2.arff", 
    "spiralsquare.arff"
]


eps_params = {
    "banana.arff": 0.15,
    "Zelnik2.arff": 0.15,
    "spiralsquare.arff": 0.2
}

def traiter_dbscan(nom_fichier):
    print("-" * 40)
    print("Fichier : " + nom_fichier)
    
    path = './dataset/dataset/artificial/' + nom_fichier
    data, meta = arff.loadarff(open(path, 'r'))
    
    points = []
    for ligne in data:
        points.append([ligne[0], ligne[1]])
    
    X = np.array(points)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # On prend min_pts = 5  par défaut
    min_pts = 5
    neighbors = NearestNeighbors(n_neighbors=min_pts)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    
    # On trie les distances pour voir le coude
    distances = np.sort(distances[:, min_pts-1])
    
    plt.figure()
    plt.plot(distances)
    plt.title("Courbe K-Distance (pour trouver Epsilon) - " + nom_fichier)
    plt.xlabel("Points triés")
    plt.ylabel("Distance Epsilon")
    plt.grid(True)
    plt.show()
    
    # On récupère le paramètre qu'on a défini plus haut
    mon_eps = eps_params.get(nom_fichier, 0.2) 
    print("Application DBSCAN avec eps =", mon_eps, " et min_samples =", min_pts)
    
    db = DBSCAN(eps=mon_eps, min_samples=min_pts)
    labels = db.fit_predict(X_scaled)
    
    # On compte les clusters (sans le bruit -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_bruit = list(labels).count(-1)
    
    print("Clusters trouvés :", n_clusters)
    print("Points de bruit :", n_bruit)
    
    if n_clusters > 1:
        score = silhouette_score(X_scaled, labels)
        print("Silhouette Score :", round(score, 3))
    else:
        print("Pas assez de clusters pour calculer le score.")
        
    #  Affichage 
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=30)
    plt.title("Résultat DBSCAN : " + nom_fichier)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

for d in datasets:
    traiter_dbscan(d)