import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# --- 1. DATA PREPARATION ---
csv_path = input("CSV Path: ").strip().replace('"', '')

try:
    df = pd.read_csv(csv_path, sep=';', decimal=',').dropna()
    df.columns = df.columns.str.strip()

    # X: N(0), Torque(2), RPM(3), Temp(4) | y: Wear a(1)
    X_wear = df.iloc[:, [0, 2, 3, 4]].values
    y_wear = df.iloc[:, 1].values

    # Scaler for the Wear Model
    X_train, X_test, y_train, y_test = train_test_split(X_wear, y_wear, test_size=0.2, random_state=42)
    scaler_wear = StandardScaler()
    X_train_s = scaler_wear.fit_transform(X_train)
    X_test_s = scaler_wear.transform(X_test)

    # 1.1 Train Wear Progression Model
    model_wear = Sequential([
        Input(shape=(4,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model_wear.compile(optimizer=Adam(0.001), loss='mse')
    print(f"\nTraining Wear Model with {len(X_train)} samples...")
    model_wear.fit(X_train_s, y_train, epochs=200, batch_size=8, verbose=0)

    # --- 2. LIFE EXPECTANCY MODEL (PROGNOSTICS) ---
    # GARANTIA: Definimos N_fail como o valor máximo de ciclos para cada condição experimental
    df_failure = df.groupby([df.columns[2], df.columns[3], df.columns[4]]).agg({
        df.columns[0]: 'max'  # Captura o último ciclo (max N) de cada amostra
    }).reset_index()

    X_fail = df_failure.iloc[:, [0, 1, 2]].values # Torque, RPM, Temp
    y_fail = df_failure.iloc[:, 3].values         # Target: Ciclo máximo registado (N_fail)

    scaler_fail = StandardScaler()
    X_fail_s = scaler_fail.fit_transform(X_fail)

    # Neural Network for N_fail Prediction
    model_life = Sequential([
        Input(shape=(3,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model_life.compile(optimizer=Adam(0.001), loss='mse')

    print(f"Training Life Model (N_fail) using {len(df_failure)} experimental end-points...")
    model_life.fit(X_fail_s, y_fail, epochs=500, verbose=0)

    # --- 3. STABLE REGIME EVALUATION (0-90% Life) ---
    # Para validar, comparamos o erro apenas na zona estável
    test_df = pd.DataFrame(X_test, columns=['N', 'Torque', 'RPM', 'Temp'])
    test_df['Real_Wear'] = y_test

    # Mapear o N_fail correspondente de cada condição para o set de teste
    max_cycles_map = df_failure.copy()
    max_cycles_map.columns = ['Torque', 'RPM', 'Temp', 'N_fail']
    test_df = test_df.merge(max_cycles_map, on=['Torque', 'RPM', 'Temp'])

    # Filtrar apenas dados até 90% da falha real
    filtered_test = test_df[test_df['N'] <= 0.90 * test_df['N_fail']]
    X_test_f_s = scaler_wear.transform(filtered_test[['N', 'Torque', 'RPM', 'Temp']].values)
    y_test_f = filtered_test['Real_Wear'].values

    y_pred_f = model_wear.predict(X_test_f_s, verbose=0).flatten()
    r2_f = r2_score(y_test_f, y_pred_f)
    t_stat, p_val = stats.ttest_rel(y_test_f, y_pred_f)

    print("\n" + "="*50)
    print("   STABLE REGIME METRICS (0-90% of Life)")
    print("="*50)
    print(f"R² Score: {r2_f:.4f}")
    print(f"p-value:  {p_val:.4f}")
    print("STATUS:   " + ("Validated (No Bias)" if p_val > 0.05 else "Statistical Bias Detected"))
    print("="*50)

    # --- 4. PREDICTION ---
    print(f"\nAvailable Torques: {np.unique(df.iloc[:, 2].values)}")
    t_f = float(input("Select Torque (Nm): ").replace(',', '.'))
    temp_f = float(input("Select Temp (ºC): ").replace(',', '.'))

    # Get RPM for the condition or global average
    match = df[(df.iloc[:, 2] == t_f) & (df.iloc[:, 4] == temp_f)]
    rpm_f = match.iloc[:, 3].mean() if not match.empty else df.iloc[:, 3].mean()

    # Predict N_fail
    n_fail_pred = model_life.predict(scaler_fail.transform([[t_f, rpm_f, temp_f]]), verbose=0)[0][0]

    print(f"\n>>> PREDICTION FOR: {t_f} Nm @ {temp_f} ºC")
    print(f"Predicted End-of-Life: {n_fail_pred:.4f} (x10^5 cycles)")

    # Table of Wear Progression
    print(f"\n{'Life %':<10} | {'Cycles':<15} | {'Predicted Wear':<15}")
    print("-" * 45)
    for p in [0.25, 0.50, 0.75, 0.90, 1.0]:
        c = n_fail_pred * p
        w = model_wear.predict(scaler_wear.transform([[c, t_f, rpm_f, temp_f]]), verbose=0)[0][0]
        label = "CRITICAL" if p == 1.0 else f"{p*100:.0f}%"
        print(f"{label:<10} | {c:<15.4f} | {w:.5f} mm")

    # Plot
    c_range = np.linspace(0, n_fail_pred, 100)
    y_plot = model_wear.predict(scaler_wear.transform([[c, t_f, rpm_f, temp_f] for c in c_range]), verbose=0)
    
    plt.figure(figsize=(10, 6))
    if not match.empty:
        plt.scatter(match.iloc[:, 0], match.iloc[:, 1], color='red', label='Experimental Data', alpha=0.5)
    plt.plot(c_range, y_plot, color='blue', label='ANN Prediction', linewidth=2)
    plt.axvline(x=n_fail_pred, color='black', linestyle='--', label='Predicted Failure Point')
    plt.title(f'Gear Wear Analysis: {t_f}Nm @ {temp_f}ºC')
    plt.xlabel('Cycles ($10^5$)'); plt.ylabel('Wear $a$ (mm)'); plt.legend(); plt.grid(True); plt.show()

except Exception as e:
    print(f"\nError: {e}")