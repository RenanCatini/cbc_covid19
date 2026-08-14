
# CBCovid19 — Classificação com dados do CBC

Este repositório contém um projeto de classificação para dados CBC (Complete Blood Count) relacionados à COVID-19. O objetivo é preparar os dados, treinar vários modelos de machine learning (KNN, Random Forest, SVM, XGBoost) e comparar resultados.

## Conteúdo

- **database/**: dados brutos e processados.
	- `raw/` — dados originais: `CBCovid19EC.csv` e `fonte.txt`.
	- `processed/` — dados tratados e conjunto dividido em treino/teste.
		- `CBCovid19_Tratado.csv`
		- `splitted/` — `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`.
- **notebooks/**: notebooks Jupyter com o pipeline do projeto.
	- `01_eda.ipynb` — análise exploratória de dados.
	- `02_data_preparation.ipynb` — limpeza, tratamento e divisão dos dados.
	- `03_knn.ipynb` — treinamento e avaliação KNN.
	- `04_rf.ipynb` — Random Forest.
	- `05_svm.ipynb` — SVM.
	- `06_xgb.ipynb` — XGBoost.
- **results/**: saídas do projeto.
	- `predicoes.csv`, `probabilidades.csv`, `estatisticas.ipynb`.
- **src/**: código fonte (se aplicável).

## Requisitos

- Python >= 3.8
- Recomenda-se criar um ambiente virtual com `venv` ou `conda`.
- Principais bibliotecas usadas (exemplos): `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `jupyter`.

Crie/ative o ambiente e instale dependências (exemplo usando `venv`):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Se não houver `requirements.txt`, instale as bibliotecas principais manualmente:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
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

## Contribuição

Se quiser contribuir, abra uma issue descrevendo a sugestão ou envie um pull request com mudanças claras e testáveis. Para melhorias sugeridas:

- adicionar `requirements.txt` com dependências fixadas;
- scripts em `src/` para automatizar o treinamento e avaliação;
- pipeline reproducível (ex: `Makefile` ou `pyproject.toml`).

## Licença e contato

Inclua aqui informações de licença (se aplicável) e contato/autor do trabalho.

---

Arquivo gerado automaticamente como README inicial para o trabalho CBCovid19.

