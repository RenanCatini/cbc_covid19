
# CBCovid19: classificação a partir de hemogramas

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

## Como executar

### 1. Criar o ambiente

É recomendado usar um ambiente virtual com Python 3.10 ou superior:

```bash
python -m venv .venv
```

Ativação no Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências utilizadas pelos notebooks e pelo código-fonte:

```bash
python -m pip install --upgrade pip
python -m pip install jupyter pandas numpy scikit-learn xgboost optuna matplotlib seaborn
```

### 2. Executar os notebooks

A partir da raiz do repositório, inicie o Jupyter:

```bash
jupyter notebook
```

Execute os notebooks nesta ordem:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_data_preparation.ipynb`
3. `notebooks/03_knn.ipynb`
4. `notebooks/04_rf.ipynb`
5. `notebooks/05_svm.ipynb`
6. `notebooks/06_xgb.ipynb`
7. `results/statistics.ipynb`

Os notebooks de modelo importam as classes de `src/`, treinam o melhor modelo encontrado e salvam as predições e probabilidades em `results/`. O arquivo `statistics.ipynb` consolida os resultados para comparação.

## Notas

- Execute os notebooks a partir da raiz do projeto para que os caminhos relativos e os imports funcionem corretamente.
- Os arquivos `X_train_pca.csv` e `X_test_pca.csv` representam o cenário com redução de dimensionalidade por PCA.
- A busca de hiperparâmetros pode demorar, especialmente no XGBoost; a quantidade de tentativas é definida no notebook correspondente.


