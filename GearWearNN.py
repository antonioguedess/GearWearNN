import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Introdução do endereço do ficheiro pelo utilizador
csv_path = input("Por favor, introduza o caminho completo do ficheiro CSV: ").strip()

# Remover aspas se o utilizador tiver colado o caminho com elas
csv_path = csv_path.replace('"', '').replace("'", "")

try:
    # CORREÇÃO: sep=';' para as colunas e decimal=',' para converter números corretamente
    df = pd.read_csv(csv_path, sep=';', decimal=',', header=0)
    
    # Limpeza: remove espaços em branco nos nomes das colunas
    df.columns = df.columns.str.strip()
    
    print("\nFicheiro carregado com sucesso!")
    print("Colunas detetadas:", df.columns.tolist())
    
    # 2. Mapeamento das colunas baseado no teu Dataset:
    # Col 0: N (x10^5) | Col 1: a (mm) | Col 2: Torque (Nm) | Col 3: n (rpm) | Col 4: Temp (ºC)
    X = df.iloc[:, [0, 2, 3, 4]].values  # Inputs
    y = df.iloc[:, 1].values             # Target (Desgaste em mm)

    # Remover possíveis linhas com valores vazios (NaN)
    nan_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[nan_mask]
    y = y[nan_mask]

    # 3. Divisão e Normalização
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Arquitetura da Rede (Otimizada para regressão de desgaste)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.1), # Ajuda na generalização
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear') # Output direto em mm
    ])

    # 5. Compilação e Treino
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    print(f"\nTreinando a rede com {len(X_train)} amostras...")
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=200, 
        batch_size=8, 
        validation_split=0.1, 
        verbose=0
    )
    
    # 6. Avaliação Final
    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTreino concluído.")
    print(f"Erro Médio Absoluto (MAE) no Teste: {mae:.5f} mm")

    # 7. Teste Interativo
    print("\n--- Simulação de Previsão ---")
    try:
        c = float(input("Introduza Nº Ciclos (x10^5): ").replace(',', '.'))
        b = float(input("Introduza Binário (Nm): ").replace(',', '.'))
        v = float(input("Introduza Velocidade (rpm): ").replace(',', '.'))
        t = float(input("Introduza Temperatura (ºC): ").replace(',', '.'))
        
        nova_amostra = scaler.transform([[c, b, v, t]])
        previsao = model.predict(nova_amostra, verbose=0)
        
        print(f"\n>>> Desgaste previsto para {c}x10^5 ciclos: {previsao[0][0]:.6f} mm")
    except ValueError:
        print("Erro: Introduza apenas valores numéricos.")

except FileNotFoundError:
    print("\nErro: O ficheiro não foi encontrado. Verifique se o caminho está correto.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")
    # Útil para debugging:
    import traceback
    traceback.print_exc()