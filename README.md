
# Classificação a partir de hemogramas

Projeto de aprendizado de máquina para classificar pacientes em relação à COVID-19 usando dados de hemograma completo (CBC). O fluxo inclui exploração dos dados, preparação das variáveis, otimização de hiperparâmetros com Optuna e comparação de quatro classificadores.

> Este projeto tem finalidade acadêmica e experimental. Seus resultados não devem ser usados para diagnóstico ou decisão clínica.

## Modelos

- K-Nearest Neighbors (KNN)
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

Os modelos são avaliados no conjunto de teste usando acurácia, precisão, recall, F1-score, coeficiente de correlação de Matthews (MCC), AUC e matriz de confusão. A seleção dos hiperparâmetros usa validação cruzada com três partições e acurácia como métrica de otimização.

## Dados

O conjunto de dados original está em `database/raw/CBCovid19EC.csv`. A fonte e as referências utilizadas estão registradas em `database/raw/fonte.txt`:

- [Dataset CBCovid19 no Mendeley Data](https://data.mendeley.com/datasets/7bmfgkkm3z/3)
- [Artigo associado](https://www.sciencedirect.com/science/article/pii/S1567134822000259)

Os arquivos tratados e as divisões de treino e teste ficam em `database/processed/`. As classes são representadas por `0` (saudável) e `1` (doente), conforme a implementação da avaliação em `src/utils.py`.

## Estrutura do repositório

```text
cbc_covid19/
├── database/
│   ├── raw/
│   │   ├── CBCovid19EC.csv          # Dados brutos
│   │   └── fonte.txt                # Fontes e referências
│   └── processed/
│       ├── CBCovid19_Tratado.csv    # Dados tratados
│       └── splitted/
│           ├── X_train.csv          # Variáveis de treino
│           ├── X_test.csv           # Variáveis de teste
│           ├── X_train_pca.csv      # Treino após PCA
│           ├── X_test_pca.csv       # Teste após PCA
│           ├── y_train.csv           # Rótulos de treino
│           └── y_test.csv            # Rótulos de teste
├── notebooks/
│   ├── 01_eda.ipynb                # Análise exploratória
│   ├── 02_data_preparation.ipynb   # Limpeza, PCA e divisão dos dados
│   ├── 03_knn.ipynb                # Experimento com KNN
│   ├── 04_rf.ipynb                 # Experimento com Random Forest
│   ├── 05_svm.ipynb                # Experimento com SVM
│   └── 06_xgb.ipynb                # Experimento com XGBoost
├── results/
│   ├── predictions.csv             # Predições no conjunto de teste
│   ├── probabilities.csv           # Probabilidades da classe positiva
│   └── statistics.ipynb            # Comparação dos resultados
├── src/
│   ├── base_experiment.py          # Fluxo comum de otimização e treino
│   ├── knn_experiment.py           # Implementação do KNN
│   ├── rf_experiment.py            # Implementação do Random Forest
│   ├── svm_experiment.py           # Implementação do SVM
│   ├── xgb_experiment.py           # Implementação do XGBoost
│   └── utils.py                    # Avaliação e persistência dos resultados
└── README.md
```
