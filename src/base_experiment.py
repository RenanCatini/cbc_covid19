import optuna

'''
    Classe Pai: Controi principais Métodos que serão usados.
'''
class BaseExperiment:
    # Contrutor da Classe Pai
    def __init__(self, X_train, X_test, y_train, y_test, scenario_name, path):
        self.X_train       = X_train
        self.X_test        = X_test
        self.y_train       = y_train
        self.y_test        = y_test
        self.scenario_name = scenario_name
        self.path          = path
        self.best_model    = None  

        
    # --- Métodos Abstratos ---        
    def _objective(self, trial):
        raise NotImplementedError("Defina a função de Objetivo para a busca dos melhores Hiperparâmetros.")

    def _build_model(self, params):
        raise NotImplementedError("Defina o modelo que será treinado.")

    # --- Métodos Globais ---
    '''
        Otimiza hiperparâmetros via Optuna e treina o melhor modelo na base de treino.

        Parâmetros:
            n_trials (int): Número de tentativas de otimização a serem executadas.
    '''
    def tune_and_fit(self, n_trials):
        print(f'Iniciando otimização do cenário {self.scenario_name}.')

        # Criando estudo e buscando melhores Hiperparâmetros
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials, show_progress_bar=True)

        print(f"Melhor pontuação (CV): {study.best_value:.4f}")
        print("Melhores parâmetros:")
        print(study.best_params)

        # Treinar e salvar melhor modelo
        self.best_model = self._build_model(self.study.best_params)
        self.best_model.fit(self.X_train, self.y_train)
        print('Modelo treinado com sucesso!')
