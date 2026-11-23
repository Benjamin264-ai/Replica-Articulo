"""
Replicacion del flujo metodologico de:
Gramegna & Giudici (2021), "SHAP and LIME: An Evaluation of Discriminative Power in Credit Risk"

Instalacion (en tu maquina, con internet):
    pip install numpy pandas scikit-learn xgboost shap lime scipy matplotlib

Uso:
    python 02_pipeline_replicacion_shap_lime.py            # corrida rapida (recomendada para probar)
    python 02_pipeline_replicacion_shap_lime.py --full     # corrida completa como el articulo (LENTA)

Pasos (numeracion = diagrama del workflow):
  1. Carga de datos
  2. (opcional) Balanceo por clustering-undersampling (Lin et al., 2017)
  3. Split estratificado 70/30
  4. Entrenar XGBoost (caja negra)
  5. Calcular SHAP y LIME sobre el set de TEST  -> dos matrices N x 46
  6. Clustering no supervisado (K-means y Espectral) sobre cada matriz, K = 2..9
  7. Evaluacion: Silhouette + Davies-Bouldin
  8. Aprendizaje supervisado: Random Forest sobre cada matriz, 50 repeticiones, AUROC
  9. Test estadistico de diferencia de medias de AUC
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (davies_bouldin_score, roc_auc_score,
                             silhouette_samples, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true", help="replica a escala completa (lento)")
ap.add_argument("--csv", default="pymes_italia_sintetico.csv")
args = ap.parse_args()

SEED = 42
TARGET = "default"
DROP_COLS = ["id_empresa", "anio"]

if args.full:
    N_EXPLAIN = None        # explica TODO el test (~41,700 obs). LIME tardara horas.
    N_SPECTRAL = 10_000     # el espectral completo (N x N) es inviable en memoria
    N_SIL = 10_000          # muestra para calcular silhouette
    N_REPS = 50             # repeticiones de Random Forest (como el articulo)
else:
    N_EXPLAIN = 3_000       # submuestra del test para que LIME termine en minutos
    N_SPECTRAL = 3_000
    N_SIL = 3_000
    N_REPS = 10

K_RANGE = range(2, 10)      # K de 2 a 9 (como el articulo)


def tic(msg):
    print(f"\n>>> {msg}")
    return time.time()


# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------------------
t = tic("1. Carga de datos")
df = pd.read_csv(args.csv)
features = [c for c in df.columns if c not in DROP_COLS + [TARGET]]
print(f"   {len(df):,} filas | {len(features)} variables | defaults = {df[TARGET].sum():,} "
      f"({df[TARGET].mean():.1%})")


# ---------------------------------------------------------------------------
# 2. BALANCEO (solo si partes de los datos crudos con desbalance ~100:1)
#    Con el CSV sintetico NO hace falta (ya viene ~4:1). Se deja la funcion para
#    cuando tengas datos reales.
# ---------------------------------------------------------------------------
def balancear_por_clustering(df_raw, n_sample_por_anio=10_000, n_clusters_por_anio=5_000):
    """
    Procedimiento descrito en el articulo (Seccion 3.1), por cada anio:
      - conservar TODOS los defaults
      - de los no-default: tomar ~10,000 al azar tal cual
      - con el resto, formar 5,000 clusters (K-means) y usar los MEDOIDES como observaciones
    """
    from sklearn.cluster import MiniBatchKMeans
    out = []
    for anio, g in df_raw.groupby("anio"):
        d = g[g[TARGET] == 1]
        nd = g[g[TARGET] == 0]
        muestra = nd.sample(n=min(n_sample_por_anio, len(nd)), random_state=SEED)
        resto = nd.drop(muestra.index)
        Xr = StandardScaler().fit_transform(resto[features])
        km = MiniBatchKMeans(n_clusters=min(n_clusters_por_anio, len(resto)),
                             random_state=SEED, n_init=1).fit(Xr)
        # medoide = observacion real mas cercana a cada centroide
        medoides = []
        for k in range(km.n_clusters):
            idx = np.where(km.labels_ == k)[0]
            if len(idx):
                dist = ((Xr[idx] - km.cluster_centers_[k]) ** 2).sum(1)
                medoides.append(resto.iloc[idx[dist.argmin()]])
        out.append(pd.concat([d, muestra, pd.DataFrame(medoides)]))
    return pd.concat(out).reset_index(drop=True)


# df = balancear_por_clustering(df)   # <- descomenta si usas datos reales crudos

# ---------------------------------------------------------------------------
# 3. SPLIT ESTRATIFICADO 70 / 30
# ---------------------------------------------------------------------------
t = tic("3. Split estratificado 70/30")
X = df[features]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
print(f"   train = {len(X_train):,} | test = {len(X_test):,}")

# ---------------------------------------------------------------------------
# 4. XGBOOST (la caja negra a explicar)
# ---------------------------------------------------------------------------
t = tic("4. Entrenando XGBoost")
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="auc", random_state=SEED, n_jobs=-1)
xgb.fit(X_train, y_train)
auc_xgb = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
print(f"   AUC XGBoost (test) = {auc_xgb:.4f}   [{time.time() - t:.0f}s]")

# Submuestra de test a explicar (en --full se explica todo)
if N_EXPLAIN is not None and N_EXPLAIN < len(X_test):
    X_exp = X_test.sample(n=N_EXPLAIN, random_state=SEED)
else:
    X_exp = X_test
y_exp = y_test.loc[X_exp.index].values

# ---------------------------------------------------------------------------
# 5. EXPLICACIONES LOCALES: SHAP y LIME  -> 2 matrices (n_obs x 46)
# ---------------------------------------------------------------------------
t = tic(f"5a. SHAP (TreeExplainer) sobre {len(X_exp):,} observaciones")
import shap

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_exp)                 # (n, 46)
if isinstance(shap_values, list):                          # compatibilidad entre versiones
    shap_values = shap_values[1]
shap_values = np.asarray(shap_values)
print(f"   matriz SHAP: {shap_values.shape}   [{time.time() - t:.0f}s]")

t = tic(f"5b. LIME (tabular) sobre {len(X_exp):,} observaciones  (puede tardar)")
from lime.lime_tabular import LimeTabularExplainer

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=features,
    class_names=["no_default", "default"],
    mode="classification",
    discretize_continuous=False,   # asi LIME entrega un peso lineal por variable original
    random_state=SEED)

lime_weights = np.zeros((len(X_exp), len(features)))
for i, row in enumerate(X_exp.values):
    exp = lime_explainer.explain_instance(
        row, xgb.predict_proba, num_features=len(features), num_samples=2000)
    for j, w in exp.as_map()[1]:                           # clase 1 = default
        lime_weights[i, j] = w
    if (i + 1) % 500 == 0:
        print(f"   LIME {i + 1:,}/{len(X_exp):,}")
print(f"   matriz LIME: {lime_weights.shape}   [{time.time() - t:.0f}s]")

# ---------------------------------------------------------------------------
# 6-7. CLUSTERING NO SUPERVISADO + METRICAS INTERNAS
#      Distancia euclidiana estandarizada (Ec. 4) == estandarizar columnas
#      (dividir por la desviacion estandar) y usar distancia euclidiana.
# ---------------------------------------------------------------------------
def evaluar_clustering(M, nombre):
    Z = StandardScaler().fit_transform(M)
    rng = np.random.default_rng(SEED)
    idx_sil = rng.choice(len(Z), size=min(N_SIL, len(Z)), replace=False)
    idx_sp = rng.choice(len(Z), size=min(N_SPECTRAL, len(Z)), replace=False)

    # --- 6a. Seleccion de K por silhouette (K = 2..9), K-means
    print(f"\n   [{nombre}] Silhouette promedio por K (K-means):")
    sil_por_k = {}
    for k in K_RANGE:
        lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Z)
        sil_por_k[k] = silhouette_score(Z[idx_sil], lab[idx_sil])
        print(f"      K={k}: {sil_por_k[k]:.3f}")
    k_best = max(sil_por_k, key=sil_por_k.get)
    print(f"   -> K optimo = {k_best}")

    # --- 6b. K-means con K = 2 (coherente con default / no default)
    lab_km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit_predict(Z)
    # --- 6c. Clustering espectral (Ng, Jordan & Weiss, 2001) sobre submuestra
    lab_sp = SpectralClustering(
        n_clusters=2, affinity="nearest_neighbors", n_neighbors=15,
        assign_labels="kmeans", random_state=SEED).fit_predict(Z[idx_sp])

    # --- 7. Metricas
    res = {
        "KMeans_Silhouette": silhouette_score(Z[idx_sil], lab_km[idx_sil]),
        "Spectral_Silhouette": silhouette_score(Z[idx_sp], lab_sp),
        "KMeans_DBI": davies_bouldin_score(Z, lab_km),
        "Spectral_DBI": davies_bouldin_score(Z[idx_sp], lab_sp),
        "K_optimo": k_best,
    }
    return res, lab_km


res_shap, lab_shap = evaluar_clustering(shap_values, "SHAP")
res_lime, lab_lime = evaluar_clustering(lime_weights, "LIME")

tabla1 = pd.DataFrame({"LIME": res_lime, "SHAP": res_shap}).round(3)
print("\n=== TABLA 1 (equivalente) - Evaluacion del clustering ===")
print(tabla1)
tabla1.to_csv("resultados_tabla1_clustering.csv")

# ---------------------------------------------------------------------------
# 8. SUPERVISADO: Random Forest sobre los pesos XAI -> AUROC (N_REPS repeticiones)
# ---------------------------------------------------------------------------
def auc_repetidos(M, y_vec, n_reps):
    aucs, curves = [], None
    for r in range(n_reps):
        Xa, Xb, ya, yb = train_test_split(M, y_vec, test_size=0.30,
                                          stratify=y_vec, random_state=r)
        rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=3,
                                    n_jobs=-1, random_state=r).fit(Xa, ya)
        p = rf.predict_proba(Xb)[:, 1]
        aucs.append(roc_auc_score(yb, p))
        if r == 0:
            curves = (yb, p)          # guardamos una corrida para la curva ROC
    return np.array(aucs), curves


t = tic(f"8. Random Forest sobre pesos XAI ({N_REPS} repeticiones)")
auc_shap, curve_shap = auc_repetidos(shap_values, y_exp, N_REPS)
auc_lime, curve_lime = auc_repetidos(lime_weights, y_exp, N_REPS)
print(f"   AUC medio SHAP = {auc_shap.mean():.4f} (+-{auc_shap.std():.4f})")
print(f"   AUC medio LIME = {auc_lime.mean():.4f} (+-{auc_lime.std():.4f})")

# ---------------------------------------------------------------------------
# 9. TEST ESTADISTICO (diferencia de medias)
# ---------------------------------------------------------------------------
t_stat, p_val = stats.ttest_ind(auc_shap, auc_lime, equal_var=False)
print(f"\n=== Test t (Welch): t = {t_stat:.3f} | p-valor = {p_val:.4g} ===")
print("   (el articulo reporta p = 0.0035 con AUC 0.864 vs 0.839)")

pd.DataFrame({"auc_shap": auc_shap, "auc_lime": auc_lime}).to_csv("resultados_auc_repeticiones.csv",
                                                                  index=False)

# ---------------------------------------------------------------------------
# GRAFICOS (equivalentes a Figura 2 y Figura 3 del articulo)
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(7, 6))
for (yy, pp), nm, c in [(curve_lime, f"RF LIME, AUC={auc_lime.mean():.3f}", "tab:blue"),
                        (curve_shap, f"RF SHAP, AUC={auc_shap.mean():.3f}", "tab:orange")]:
    fpr, tpr, _ = roc_curve(yy, pp)
    ax.plot(fpr, tpr, label=nm, color=c)
ax.plot([0, 1], [0, 1], "--", color="blue", label="No skill")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(); ax.set_title("Curvas ROC (equivalente a Fig. 3)")
fig.savefig("fig_roc_lime_vs_shap.png", dpi=150, bbox_inches="tight")

# Proyeccion 2D con PCA coloreada por cluster (equivalente a Fig. 2)
from sklearn.decomposition import PCA

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for a, M, lab, nm in [(axs[0], lime_weights, lab_lime, "LIME"),
                      (axs[1], shap_values, lab_shap, "SHAP")]:
    P = PCA(2, random_state=SEED).fit_transform(StandardScaler().fit_transform(M))
    a.scatter(P[:, 0], P[:, 1], c=lab, s=4, cmap="viridis")
    a.set_title(f"{nm}: clusters (K-means, K=2) en 2D")
fig.savefig("fig_clusters_2d.png", dpi=150, bbox_inches="tight")
print("\nArchivos generados: resultados_tabla1_clustering.csv, resultados_auc_repeticiones.csv, "
      "fig_roc_lime_vs_shap.png, fig_clusters_2d.png")
