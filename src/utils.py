import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             matthews_corrcoef, confusion_matrix)
import os
import pandas as pd

'''
    Calcula e exibe as métricas de desempenho e a matriz de confusão para
    um conjunto de previsões do modelo.

    Parâmetros:
        y_test (array-like): Valores reais do conjunto de teste.
        y_pred (array-like): Classes previstas pelo modelo.
        y_prob (array-like): Probabilidades associadas à classe positiva.
        scenario_name (str): Nome do cenário/modelo para identificar a saída.
'''
def evaluate_model(y_test, y_pred, y_prob, scenario_name):
    print(f"=== Métricas {scenario_name}  ===")
    print(f"    Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"    Precisão: {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"    Recall:   {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"    F1-Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"    MCC:      {matthews_corrcoef(y_test, y_pred):.4f}")
    print(f"    AUC:      {roc_auc_score(y_test, y_prob):.4f}")

    # Plotar Matriz de Confusão
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)

    # Labels ajustados exclusivamente para as duas doenças
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Saudável (0)', 'Doente (1)'],
                yticklabels=['Saudável (0)', 'Doente (1)'])

    plt.title(f'Matriz de Confusão - {scenario_name}')
    plt.xlabel('Previsto pelo Modelo')
    plt.ylabel('Verdadeiro (Real)')
    plt.show()

'''
    Salva as previsões e as probabilidades de teste em arquivos CSV dentro da
    pasta de resultados do projeto.

    Parâmetros:
        y_pred (array-like): Classes previstas para o conjunto de teste.
        y_prob (array-like): Probabilidades previstas para a classe positiva.
        scenario_name (str): Nome do cenário/modelo usado no registro do arquivo.
        path (str): Caminho da pasta raiz do projeto.
'''
def save_results(y_pred, y_prob, scenario_name, path):
    # Encontrar pasta de resultados
    pasta_results = os.path.join(path, 'results')
    os.makedirs(pasta_results, exist_ok=True)

    arquivo_pred = os.path.join(pasta_results, 'predictions.csv')
    arquivo_proba = os.path.join(pasta_results, 'probabilities.csv')

    # Monta a linha com as predições de teste
    linha_pred_teste = [scenario_name, 'teste'] + list(y_pred)

    # Salva adicionando no final do arquivo (mode='a')
    pd.DataFrame([linha_pred_teste]).to_csv(
        arquivo_pred, mode='a', header=False, index=False
    )

    # Monta a linha com as probabilidades de teste 
    probs = (
        y_prob[:, 1]
        if hasattr(y_prob, 'ndim') and y_prob.ndim == 2
        else y_prob
    )
    linha_proba_teste = [scenario_name, 'teste'] + list(probs)

    pd.DataFrame([linha_proba_teste]).to_csv(
        arquivo_proba, mode='a', header=False, index=False
    )