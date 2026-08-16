
# CBCovid19 — Classificação de dados de hemograma para COVID-19

Este projeto tem como objetivo classificar pacientes com base em dados de hemograma (CBC) para distinguir casos positivos e negativos de COVID-19. A abordagem combina preparação dos dados, busca de hiperparâmetros e comparação de modelos de machine learning.

## Modelos incluídos

- KNN
- Random Forest
- SVM
- XGBoost

A otimização de hiperparâmetros é feita com Optuna e a avaliação dos modelos considera métricas como acurácia, precisão, recall, F1-score, MCC e AUC.

## Estrutura do repositório

```text
cbc_covid19/
├── database/
│   ├── raw/
│   │   ├── CBCovid19EC.csv          # Dados brutos originais
│   │   └── fonte.txt                # Fonte e metadados do conjunto
│   └── processed/
│       ├── CBCovid19_Tratado.csv    # Dataset tratado
│       └── splitted/
│           ├── X_train.csv
│           ├── X_test.csv
│           ├── y_train.csv
│           └── y_test.csv
├── notebooks/
│   ├── 01_eda.ipynb                # Exploração inicial dos dados
│   ├── 02_data_preparation.ipynb   # Limpeza, tratamento e divisão em treino/teste
│   ├── 03_knn.ipynb                # Experimento com KNN
│   ├── 04_rf.ipynb                 # Experimento com Random Forest
│   ├── 05_svm.ipynb                # Experimento com SVM
│   └── 06_xgb.ipynb               # Experimento com XGBoost
├── results/
│   ├── predictions.csv             # Predições dos modelos
│   ├── probabilities.csv          # Probabilidades estimadas
│   └── statistics.ipynb           # Consolidação e análise dos resultados
├── src/
│   ├── base_experiment.py         # Classe base para otimização e treinamento
│   ├── knn_experiment.py          # Implementação do modelo KNN
│   ├── rf_experiment.py           # Implementação do modelo Random Forest
│   ├── svm_experiment.py          # Implementação do modelo SVM
│   ├── xgb_experiment.py          # Implementação do modelo XGBoost
│   └── utils.py                  # Funções de avaliação e salvamento de resultados
├── README.md
├── .gitignore
└── requirements.txt               # Dependências do projeto (se presente no ambiente)
```

## Fluxo do projeto

1. Exploração e análise dos dados em `notebooks/01_eda.ipynb`.
2. Preparação dos dados e geração dos arquivos de treino/teste em `notebooks/02_data_preparation.ipynb`.
3. Treinamento e comparação dos classificadores nos notebooks numerados de `03` a `06`.
4. Armazenamento dos resultados em `results/`.

## Como executar

A forma mais direta é abrir o projeto no Jupyter Notebook e executar os notebooks na ordem:

```bash
jupyter notebook
```

Depois, execute:

- `01_eda.ipynb`
- `02_data_preparation.ipynb`
- `03_knn.ipynb`
- `04_rf.ipynb`
- `05_svm.ipynb`
- `06_xgb.ipynb`

## Observações

- Os dados processados já estão organizados em pastas e arquivos CSV prontos para uso.
- O projeto utiliza uma estrutura modular em `src/` para reutilizar a lógica de otimização e avaliação entre diferentes modelos.
- O arquivo `results/statistics.ipynb` pode ser usado para consolidar as métricas e comparar o desempenho dos classificadores.

## Dependências principais

- Python
- pandas
- numpy
- scikit-learn
- xgboost
- optuna
- matplotlib
- seaborn

Se necessário, a instalação pode ser feita com:

```bash
pip install -r requirements.txt
```


