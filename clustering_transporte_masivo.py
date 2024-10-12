import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# --------------- Carga y Preparación de Datos ---------------

# Cargar el dataset
df = pd.read_csv('transporte_masivo_ampliado.csv')

# Selección de características para clustering
X = df[['numero_pasajeros', 'temperatura', 'lluvia', 'evento_especial']]

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------- Determinación del Número Óptimo de Clusters ---------------

inertia = []
silhouette_scores = []
K = range(2, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Visualización del Método del Codo
plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'bo-')
plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.xticks(K)
plt.show()

# Visualización de Silhouette Scores
plt.figure(figsize=(8,5))
plt.plot(K, silhouette_scores, 'bo-')
plt.title('Silhouette Score para Diferentes Números de Clusters')
plt.xlabel('Número de Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(K)
plt.show()

# Basado en el Método del Codo y Silhouette Score, seleccionar el número óptimo de clusters
# Por ejemplo, si el codo está en k=3 y el Silhouette Score es alto
optimal_k = 3

# --------------- Entrenamiento del Modelo K-Means con k Óptimo ---------------

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)

# Asignación de etiquetas de cluster al DataFrame original
df['cluster'] = kmeans.labels_

# --------------- Evaluación del Modelo ---------------

# Cálculo del Silhouette Score
score = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette Score para k={optimal_k}: {score}")

# --------------- Visualización de Clusters ---------------

# Gráfico de dispersión: Temperatura vs Número de Pasajeros
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='temperatura', y='numero_pasajeros', hue='cluster', palette='viridis')
plt.title('Clusters de Estaciones Basados en Temperatura y Número de Pasajeros')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Número de Pasajeros')
plt.legend(title='Cluster')
plt.show()

# Gráfico de dispersión: Lluvia vs Número de Pasajeros
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='lluvia', y='numero_pasajeros', hue='cluster', palette='viridis')
plt.title('Clusters de Estaciones Basados en Lluvia y Número de Pasajeros')
plt.xlabel('Lluvia (mm)')
plt.ylabel('Número de Pasajeros')
plt.legend(title='Cluster')
plt.show()

# --------------- Importancia de las Características ---------------

# Aunque K-Means no proporciona directamente la importancia de las características,
# podemos interpretar qué características influyen más en la formación de clusters.

# Opcional: Análisis de los Centroides
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids, columns=['numero_pasajeros', 'temperatura', 'lluvia', 'evento_especial'])
centroids_df['cluster'] = range(optimal_k)

print("Centroides de los Clusters:")
print(centroids_df)

# --------------- Guardar Resultados ---------------

# Guardar el DataFrame con la asignación de clusters
df.to_csv('transporte_masivo_con_clusters.csv', index=False)
print("Dataset con clusters guardado como 'transporte_masivo_con_clusters.csv'.")
