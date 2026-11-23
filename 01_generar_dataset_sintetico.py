"""
Dataset SINTETICO que replica la estructura descrita en:
Gramegna & Giudici (2021), "SHAP and LIME: An Evaluation of Discriminative Power
in Credit Risk", Front. Artif. Intell. 4:752558.

Lo que el articulo SI especifica (se respeta tal cual):
  - ~139,000 observaciones de PYMEs italianas
  - 27,200 defaults  (ratio no-default : default ~ 5 : 1)
  - 46 variables explicativas (el articulo dice "46 dimensiones")
  - 6 anios de datos (se asigna una columna 'anio')
  - Variable objetivo binaria: default

Lo que el articulo NO especifica (supuestos mios, ajustalos si tu asesor tiene la lista real):
  - Los nombres de las 46 variables (el articulo no las lista). Se usan indicadores
    financieros tipicos de la base Bureau van Dijk / AIDA.
  - Distribuciones, correlaciones y la relacion variable -> default.

Uso:
    python 01_generar_dataset_sintetico.py
Salida:
    pymes_italia_sintetico.csv
"""
import numpy as np
import pandas as pd

SEED = 42
N_TOTAL = 139_000
N_DEFAULT = 27_200
ANIOS = [2014, 2015, 2016, 2017, 2018, 2019]

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1) Factores latentes (dan correlacion realista entre ratios financieros)
#    tamanio, rentabilidad, apalancamiento, liquidez, eficiencia, crecimiento
# ---------------------------------------------------------------------------
corr = np.array([
    # tam   rent  apal  liq   efic  crec
    [1.00, 0.15, 0.10, -0.10, 0.10, 0.05],
    [0.15, 1.00, -0.45, 0.25, 0.30, 0.35],
    [0.10, -0.45, 1.00, -0.50, -0.10, -0.10],
    [-0.10, 0.25, -0.50, 1.00, 0.05, 0.05],
    [0.10, 0.30, -0.10, 0.05, 1.00, 0.20],
    [0.05, 0.35, -0.10, 0.05, 0.20, 1.00],
])
L = np.linalg.cholesky(corr)
Z = rng.standard_normal((N_TOTAL, 6)) @ L.T
tam, rent, apal, liq, efic, crec = Z.T


def ruido(s=1.0):
    return rng.standard_normal(N_TOTAL) * s


def mezcla(*pares, s=0.6):
    """combinacion lineal de factores + ruido idiosincratico"""
    out = ruido(s)
    for f, w in pares:
        out = out + w * f
    return out


F = {}  # diccionario de variables

# ---------------------------------------------------------------------------
# 2) Variables (46) agrupadas por familia
# ---------------------------------------------------------------------------
# --- Tamanio / estructura (6)
F["total_activos_log"] = 13.0 + 1.2 * tam + ruido(0.3)                      # ln(EUR)
F["ingresos_log"] = F["total_activos_log"] - 0.2 + 0.4 * efic + ruido(0.4)  # ln(EUR)
F["empleados_log"] = 2.3 + 0.7 * tam + ruido(0.5)
F["edad_empresa"] = np.clip(rng.gamma(3.0, 6.0, N_TOTAL) + 1, 1, 80).round()  # anios
F["activo_fijo_sobre_activos"] = np.clip(0.30 + 0.12 * tam - 0.05 * liq + ruido(0.12), 0, 1)
F["intangibles_sobre_activos"] = np.clip(np.abs(0.04 + 0.03 * tam + ruido(0.05)), 0, 0.6)

# --- Rentabilidad (8)
F["roa"] = 0.03 + 0.05 * rent + ruido(0.025)
F["roe"] = 0.06 + 0.12 * rent - 0.03 * apal + ruido(0.06)
F["roi"] = 0.05 + 0.06 * rent + ruido(0.03)
F["margen_ebitda"] = 0.10 + 0.06 * rent + ruido(0.03)
F["margen_neto"] = 0.03 + 0.04 * rent + ruido(0.02)
F["margen_operativo"] = 0.06 + 0.05 * rent + ruido(0.025)
F["ebit_sobre_activos"] = 0.045 + 0.05 * rent + ruido(0.025)
F["utilidad_retenida_sobre_activos"] = 0.10 + 0.10 * rent - 0.04 * apal + ruido(0.06)

# --- Liquidez (6)
F["razon_corriente"] = np.clip(np.exp(0.35 + 0.35 * liq + ruido(0.25)), 0.1, 15)
F["prueba_acida"] = np.clip(F["razon_corriente"] * np.clip(0.75 + ruido(0.1), 0.3, 1), 0.05, 12)
F["razon_efectivo"] = np.clip(np.exp(-1.3 + 0.45 * liq + ruido(0.5)), 0.001, 8)
F["capital_trabajo_sobre_activos"] = 0.15 + 0.12 * liq - 0.05 * apal + ruido(0.08)
F["efectivo_sobre_activos"] = np.clip(0.07 + 0.04 * liq + ruido(0.04), 0, 0.8)
F["flujo_caja_sobre_deuda"] = 0.12 + 0.08 * rent + 0.04 * liq + ruido(0.07)

# --- Apalancamiento / estructura financiera (8)
F["deuda_sobre_activos"] = np.clip(0.60 + 0.15 * apal + ruido(0.08), 0.02, 1.8)
F["deuda_sobre_patrimonio"] = np.clip(np.exp(0.6 + 0.6 * apal + ruido(0.35)), 0.01, 60)
F["patrimonio_sobre_activos"] = 1 - np.clip(F["deuda_sobre_activos"], 0, 1.5) + ruido(0.02)
F["deuda_lp_sobre_activos"] = np.clip(0.18 + 0.08 * apal + ruido(0.08), 0, 1)
F["deuda_cp_sobre_activos"] = np.clip(0.38 + 0.10 * apal - 0.05 * liq + ruido(0.08), 0, 1.5)
F["deuda_bancaria_sobre_deuda"] = np.clip(0.35 + 0.08 * apal + ruido(0.15), 0, 1)
F["apalancamiento_financiero"] = 1.0 / np.clip(F["patrimonio_sobre_activos"], 0.03, 1.2)
F["deuda_neta_sobre_ebitda"] = np.clip(
    2.5 + 1.2 * apal - 0.8 * rent + ruido(1.0), -5, 40)

# --- Cobertura de deuda (4)
F["cobertura_intereses"] = np.clip(np.exp(1.4 + 0.6 * rent - 0.4 * apal + ruido(0.5)), 0.05, 200)
F["ebitda_sobre_gastos_fin"] = F["cobertura_intereses"] * np.clip(1.25 + ruido(0.15), 0.6, 2)
F["gastos_fin_sobre_ingresos"] = np.clip(0.02 + 0.015 * apal - 0.008 * rent + ruido(0.008), 0, 0.5)
F["dscr"] = np.clip(np.exp(0.2 + 0.4 * rent - 0.3 * apal + ruido(0.35)), 0.05, 30)

# --- Actividad / eficiencia (5)
F["rotacion_activos"] = np.clip(1.1 + 0.35 * efic + ruido(0.25), 0.05, 8)
F["dias_cobro"] = np.clip(80 - 20 * efic - 8 * liq + ruido(18), 1, 400)
F["dias_pago"] = np.clip(90 - 10 * efic + 18 * apal + ruido(20), 1, 500)
F["dias_inventario"] = np.clip(60 - 10 * efic + ruido(25), 0, 365)
F["rotacion_capital_trabajo"] = np.clip(5 + 2 * efic - 1.0 * liq + ruido(3), -30, 60)

# --- Crecimiento (3)
F["crec_ingresos"] = 0.03 + 0.08 * crec + ruido(0.08)
F["crec_activos"] = 0.03 + 0.06 * crec + ruido(0.07)
F["crec_utilidad"] = 0.02 + 0.30 * crec + 0.1 * rent + ruido(0.30)

# --- Categoricas codificadas (6) -> el articulo trabaja con 46 dimensiones numericas
F["sector_codigo"] = rng.choice(np.arange(1, 11), N_TOTAL,
                                p=[.22, .17, .14, .11, .09, .08, .06, .06, .04, .03])
F["region_codigo"] = rng.choice(np.arange(1, 21), N_TOTAL)     # 20 regiones italianas
F["forma_juridica_codigo"] = rng.choice([1, 2, 3, 4], N_TOTAL, p=[.62, .25, .08, .05])
F["es_exportadora"] = rng.binomial(1, np.clip(0.25 + 0.1 * tam, 0.02, 0.9))
F["pertenece_grupo"] = rng.binomial(1, np.clip(0.18 + 0.08 * tam, 0.02, 0.8))
F["num_bancos"] = np.clip(np.round(2 + 0.8 * tam + ruido(1.0)), 1, 12)

assert len(F) == 46, f"Se esperaban 46 variables, hay {len(F)}"

X = pd.DataFrame(F)

# ---------------------------------------------------------------------------
# 3) Probabilidad de default (logistica con no linealidades e interacciones,
#    para que un modelo de arboles tenga ventaja sobre uno lineal)
# ---------------------------------------------------------------------------
logit = (
    -4.0 * X["roa"] / 0.05
    + 0.9 * (X["deuda_sobre_activos"] - 0.6) / 0.15
    - 0.8 * np.log(X["razon_corriente"]) / 0.35
    - 0.7 * np.log(X["cobertura_intereses"]) / 0.8
    + 0.5 * (X["dias_pago"] - 90) / 25
    - 0.35 * (X["total_activos_log"] - 13) / 1.2
    - 0.25 * np.log(X["edad_empresa"])
    - 0.6 * X["crec_ingresos"] / 0.08
    + 1.2 * (X["deuda_cp_sobre_activos"] > 0.55)                       # umbral
    + 1.8 * ((X["roa"] < 0) & (X["deuda_sobre_activos"] > 0.7))        # interaccion
    + 1.4 * ((X["razon_corriente"] < 0.8) & (X["dias_cobro"] > 100))   # interaccion
    + 1.0 * ((X["cobertura_intereses"] < 1.5) & (X["deuda_neta_sobre_ebitda"] > 4))  # interaccion
    + 0.15 * (X["sector_codigo"].isin([3, 7])).astype(int)             # sectores de riesgo
    + rng.standard_normal(N_TOTAL) * 0.45                               # ruido no explicado
)
# estandarizar para que el calibrado de abajo sea estable
logit = (logit - logit.mean()) / logit.std()


def calibrar_intercepto(z, objetivo):
    """busca el intercepto para que la suma de probabilidades = objetivo"""
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        s = (1 / (1 + np.exp(-(2.6 * z + mid)))).sum()
        lo, hi = (mid, hi) if s < objetivo else (lo, mid)
    return (lo + hi) / 2


b0 = calibrar_intercepto(logit.values, N_DEFAULT)
p = 1 / (1 + np.exp(-(2.6 * logit.values + b0)))

# Se eligen EXACTAMENTE 27,200 defaults, ponderando por p (sin reemplazo)
idx_def = rng.choice(N_TOTAL, size=N_DEFAULT, replace=False, p=p / p.sum())
y = np.zeros(N_TOTAL, dtype=int)
y[idx_def] = 1

# ---------------------------------------------------------------------------
# 4) Identificadores, anio y exportacion
# ---------------------------------------------------------------------------
df = X.copy()
df.insert(0, "id_empresa", [f"IT{str(i).zfill(7)}" for i in rng.permutation(N_TOTAL) + 1])
df.insert(1, "anio", rng.choice(ANIOS, N_TOTAL))
df["default"] = y

df = df.round(5)
df.to_csv("pymes_italia_sintetico.csv", index=False)

print("Filas:", len(df), "| Columnas:", df.shape[1],
      "(1 id + 1 anio + 46 variables + 1 target)")
print("Defaults:", int(df['default'].sum()),
      f"({df['default'].mean():.2%}) | ratio no-default:default = "
      f"{(1 - df['default'].mean()) / df['default'].mean():.2f} : 1")
print(df.groupby("anio")["default"].agg(["count", "sum"]))
