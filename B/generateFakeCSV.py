import pandas as pd
import numpy as np

def gerar_dataset_dificil(filename='dataset_raw_series.csv'):
    np.random.seed(42)
    
    n_pessoas = 5
    n_atividades = 5
    n_features = 10
    tamanho_janela = 50 
    
    data = []
    
    print("A gerar dados COMPLEXOS (Sobreposição de valores)...")
    
    for pessoa in range(1, n_pessoas + 1):
        
        # Cada pessoa tem um "estilo" ligeiramente diferente (ex: mais rápido ou mais lento)
        # Isso dificulta o teste Subject Independent
        estilo_pessoa_freq = 1.0 + (np.random.rand() * 0.2) - 0.1 # +/- 10% velocidade
        estilo_pessoa_amp  = 1.0 + (np.random.rand() * 0.2) - 0.1 # +/- 10% amplitude
        
        for ativ in range(1, n_atividades + 1):
            
            n_janelas = np.random.randint(15, 30) 
            
            for janela_idx in range(n_janelas):
                
                # Time steps
                time_steps = np.linspace(0, 4*np.pi, tamanho_janela)
                
                # Definição das Atividades (Baseadas em Frequência/Forma, não em Offset!)
                # Todas centragens em 0, mas com comportamentos diferentes.
                
                if ativ == 1: # Ex: Estático / Parado (Baixa amplitude, muito ruído)
                    freq_base = 0.1
                    amp_base = 0.2
                    ruido_level = 0.1
                    
                elif ativ == 2: # Ex: Caminhar (Sinusoidal limpa, freq média)
                    freq_base = 1.0
                    amp_base = 1.5
                    ruido_level = 0.2
                    
                elif ativ == 3: # Ex: Correr (Freq alta, amplitude alta)
                    freq_base = 2.5
                    amp_base = 2.5
                    ruido_level = 0.5
                    
                elif ativ == 4: # Ex: Subir Escadas (Padrão complexo, soma de senos)
                    freq_base = 1.5 # Vai ser misturado com outra freq
                    amp_base = 1.8
                    ruido_level = 0.3
                    
                else: # Ex: Ativ 5 - Movimento Irregular (Frequencia variável)
                    freq_base = 0.5
                    amp_base = 1.0
                    ruido_level = 0.8 # Muito ruído
                
                # Loop pelas features
                for t, ts_val in enumerate(time_steps):
                    row = []
                    row.append(pessoa)
                    
                    for f in range(n_features):
                        # Variação sutil entre features (simula eixos X, Y, Z de sensores)
                        phase_shift = f * (np.pi / 4)
                        
                        # Aplica o estilo da pessoa
                        freq_final = freq_base * estilo_pessoa_freq
                        amp_final = amp_base * estilo_pessoa_amp
                        
                        # --- CÁLCULO DO VALOR (LÓGICA COMPLEXA) ---
                        
                        if ativ == 4: 
                            # Atividade 4 é mais complexa: Seno + Cosseno rápido
                            valor = np.sin(ts_val * freq_final + phase_shift) + \
                                    0.5 * np.cos(ts_val * freq_final * 2)
                            valor = valor * amp_final
                            
                        elif ativ == 5:
                            # Atividade 5 é caótica: Seno lento + Ruído forte
                            valor = np.sin(ts_val * freq_final + phase_shift) * amp_final
                            # O ruído será adicionado depois com mais força
                            
                        else:
                            # Atividades 1, 2, 3: Onda padrão
                            valor = np.sin(ts_val * freq_final + phase_shift) * amp_final

                        # Adicionar ruído
                        ruido = np.random.normal(0, ruido_level)
                        valor_final = valor + ruido
                        
                        row.append(round(valor_final, 5))
                    
                    # Label
                    row.append(ativ)
                    data.append(row)

    cols = ['person_id'] + [f'feat_{i}' for i in range(1, n_features + 1)] + ['activity_id']
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(filename, index=False)
    print(f"Dataset difícil gerado: '{filename}'")
    return df

# Executar
df_raw = gerar_dataset_dificil()