import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ==========================================
# 1. PARÁMETROS Y DATOS EXPERIMENTALES
# ==========================================
g = 9.81  # Aceleración de la gravedad (m/s^2)

# Datos medidos en el experimento
# Masa del objeto (kg)
masas = np.array([0.25, 0.50, 0.75, 1.00, 1.25]) 

# Fuerza de tensión registrada en el dinamómetro (N)
tensiones = np.array([0.98, 1.91, 2.90, 3.85, 4.80]) 

# Velocidad de la caminadora (m/s) en las mediciones
velocidades = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) 

# Ángulo de inclinación de la caminadora (grados)
angulo_deg = 0.0  
angulo_rad = np.radians(angulo_deg)

# ==========================================
# 2. CÁLCULOS DINÁMICOS Y FÍSICOS
# ==========================================
# Fuerza Normal: N = m * g * cos(theta)
normales = masas * g * np.cos(angulo_rad)

# Fuerza de Fricción: En estado estacionario, f_k = T - m * g * sin(theta)
fuerzas_friccion = tensiones - (masas * g * np.sin(angulo_rad))

# Coeficiente de fricción puntual: mu_k = f_k / N
mu_k_puntual = fuerzas_friccion / normales

# Estadísticas del coeficiente
mu_k_promedio = np.mean(mu_k_puntual)
mu_k_std = np.std(mu_k_puntual)

# Regresión lineal para T vs N (Pendiente = mu_k)
res = linregress(normales, fuerzas_friccion)
mu_k_regresion = res.slope
r_cuadrado = res.rvalue**2

# ==========================================
# 3. MOSTRAR RESULTADOS EN CONSOLA
# ==========================================
print("==================================================")
print("     ANÁLISIS DINÁMICO DE LA CAMINADORA")
print("==================================================")
for i in range(len(masas)):
    print(f"Ensayo {i+1}: Masa = {masas[i]:.2f} kg | N = {normales[i]:.2f} N | T = {tensiones[i]:.2f} N | μk = {mu_k_puntual[i]:.3f}")

print("--------------------------------------------------")
print(f"Coeficiente de Fricción Promedio (μk): {mu_k_promedio:.3f} ± {mu_k_std:.3f}")
print(f"Coeficiente por Regresión Lineal (μk): {mu_k_regresion:.3f}")
print(f"Coeficiente de Determinación (R²):     {r_cuadrado:.4f}")
print("==================================================")

# ==========================================
# 4. GRAFICACIÓN DE RESULTADOS
# ==========================================
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Gráfica 1: Fuerza de Fricción vs Fuerza Normal
ax[0].scatter(normales, fuerzas_friccion, color='navy', label='Datos Experimentales', zorder=3)
fit_x = np.linspace(0, max(normales)*1.1, 100)
fit_y = res.slope * fit_x + res.intercept
ax[0].plot(fit_x, fit_y, color='crimson', linestyle='--', label=f'Ajuste Lineal (μk = {mu_k_regresion:.3f})')
ax[0].set_xlabel('Fuerza Normal, $N$ (N)', fontsize=11)
ax[0].set_ylabel('Fuerza de Fricción, $f_k$ (N)', fontsize=11)
ax[0].set_title('Determinación del Coeficiente de Fricción Dynamic ($\mu_k$)', fontsize=12)
ax[0].grid(True, linestyle=':', alpha=0.6)
ax[0].legend()

# Gráfica 2: Simulación del perfil de velocidad v(t) del cuerpo al liberarse
t = np.linspace(0, 1.5, 100)
a_constante = mu_k_regresion * g  # a = μk * g
v_t = a_constante * t
v_cinta = 1.5 # m/s
v_t = np.minimum(v_t, v_cinta) # Alcanza la velocidad de la cinta

ax[1].plot(t, v_t, color='darkgreen', linewidth=2, label='Velocidad del bloque $v(t)$')
ax[1].axhline(y=v_cinta, color='gray', linestyle=':', label='Velocidad de la cinta ($v_{cinta}$)')
ax[1].set_xlabel('Tiempo, $t$ (s)', fontsize=11)
ax[1].set_ylabel('Velocidad, $v$ (m/s)', fontsize=11)
ax[1].set_title('Respuesta Cinemática al Soltar el Bloque sobre la Cinta', fontsize=12)
ax[1].grid(True, linestyle=':', alpha=0.6)
ax[1].legend()

plt.tight_layout()
plt.show()