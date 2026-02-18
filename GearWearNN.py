import os
# Silenciar avisos do TensorFlow (deve ser feito ANTES dos imports do TF)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from scipy import stats  # Para o teste t de Student

# 1. Carregamento e Preparação
csv_path = input("Caminho do CSV: ").strip().replace('"', '')

try:
    # Carregamento do Dataset (ajustado para o teu ficheiro)
    df = pd.read_csv(csv_path, sep=';', decimal=',').dropna()
    df.columns = df.columns.str.strip()
    
    # X: N(0), Torque(2), RPM(3), Temp(4) | y: a(1)
    X = df.iloc[:, [0, 2, 3, 4]].values
    y = df.iloc[:, 1].values

    # 2. Divisão e Escalonamento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 3. Rede Neuronal
    model = Sequential([
        Input(shape=(4,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    
    print(f"\nA treinar com {len(X_train)} amostras... (Aguarde)")
    model.fit(X_train_s, y_train, epochs=200, batch_size=8, verbose=0)

    # 4. Métricas e Estatística
    y_pred = model.predict(X_test_s, verbose=0).flatten()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Raiz Quadrada do MSE
    r2 = r2_score(y_test, y_pred)

    # Teste t de Student Pareado (compara as médias das previsões vs real)
    t_stat, p_val = stats.ttest_rel(y_test, y_pred)

    print("\n" + "="*40)
    print(f"       ESTATÍSTICAS DO MODELO")
    print("="*40)
    print(f"MSE:   {mse:.6f} mm²")
    print(f"RMSE:  {rmse:.6f} mm")
    print(f"R²:    {r2:.4f}")
    print(f"p-val (t-Student): {p_val:.4f}")

    if p_val > 0.05:
        print("Interpretação: Modelo validado (sem bias significativo).")
    else:
        print("Interpretação: Existe um desvio sistemático entre real/previsto.")
    print("="*40)

    # 5. Visualização Seletiva
    print(f"\nTorques disponíveis: {np.unique(df.iloc[:, 2].values)}")
    print(f"Temperaturas disponíveis: {np.unique(df.iloc[:, 4].values)}")
    
    t_f = float(input("\nTorque para gráfico (Nm): ").replace(',', '.'))
    temp_f = float(input("Temp. para gráfico (ºC): ").replace(',', '.'))

    # Filtrar dados reais e criar curva do modelo
    real = df[(df.iloc[:, 2] == t_f) & (df.iloc[:, 4] == temp_f)].sort_values(by=df.columns[0])
    
    if not real.empty:
        c_range = np.linspace(real.iloc[:, 0].min(), real.iloc[:, 0].max(), 100)
        v_avg = real.iloc[:, 3].mean()
        X_p = scaler.transform([[c, t_f, v_avg, temp_f] for c in c_range])
        y_p = model.predict(X_p, verbose=0)

        plt.figure(figsize=(10, 6))
        plt.scatter(real.iloc[:, 0], real.iloc[:, 1], color='red', label='Dados Reais (Dataset)')
        plt.plot(c_range, y_p, color='blue', label='Previsão da Rede Neuronal', linewidth=2)
        plt.title(f'Validação do Modelo: Torque {t_f}Nm @ {temp_f}ºC')
        plt.xlabel('Número de Ciclos (x10^5)')
        plt.ylabel('Desgaste a (mm)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("\nErro: Combinação de Torque/Temp não encontrada no ficheiro.")

except Exception as e:
    print(f"\nOcorreu um erro: {e}")