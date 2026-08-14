
# CBCovid19 — Classificação com dados do CBC

Este repositório contém um projeto de classificação para dados CBC (Complete Blood Count) relacionados à COVID-19. O objetivo é preparar os dados, treinar vários modelos de machine learning (KNN, Random Forest, SVM, XGBoost) e comparar resultados.

## Conteúdo

```text
cbc_covid19/
├── database/
│   ├── raw/
│   │   ├── CBCovid19EC.csv              # Dados brutos originais
│   │   └── fonte.txt                    # Informações da fonte dos dados
│   └── processed/
│       ├── CBCovid19_Tratado.csv        # Dataset limpo e tratado
│       └── splitted/                    # Divisão treino/teste escalada
│           ├── X_train.csv
│           ├── X_test.csv
│           ├── y_train.csv
│           └── y_test.csv
├── notebooks/
│   ├── 01_eda.ipynb                     # Análise exploratória de dados
│   ├── 02_data_preparation.ipynb       # Limpeza, tratamento e partição dos dados
│   ├── 03_knn.ipynb                     # Treinamento e avaliação: KNN
│   ├── 04_rf.ipynb                      # Treinamento e avaliação: Random Forest
│   ├── 05_svm.ipynb                     # Treinamento e avaliação: SVM
│   └── 06_xgb.ipynb                     # Treinamento e avaliação: XGBoost
├── results/
│   ├── predicoes.csv                    # Predições consolidadas dos modelos
│   ├── probabilidades.csv               # Probabilidades estimadas
│   └── estatisticas.ipynb               # Consolidação de métricas e curvas ROC
└── src/                                 # Módulos e scripts auxiliares
```

## Uso / Execução

1. Abra o Jupyter Notebook no diretório do projeto:

```bash
jupyter notebook
```

2. Execute os notebooks na ordem numérica para reproduzir o pipeline:

- `01_eda.ipynb` → exploração dos dados
- `02_data_preparation.ipynb` → limpeza e preparação (gera os arquivos em `database/processed/`)
- `03_knn.ipynb`, `04_rf.ipynb`, `05_svm.ipynb`, `06_xgb.ipynb` → treinamento e avaliação dos modelos

3. Resultados e previsões são armazenados em `results/`.

## Estrutura dos dados

- As colunas e o significado dos atributos estão disponíveis no arquivo de origem em `database/raw/` ou podem ser inferidos no notebook `01_eda.ipynb`.


