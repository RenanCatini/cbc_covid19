from src.base_experiment import BaseExperiment
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

class SVMExperiment(BaseExperiment):
    # Contrutor da Classe
    def __init__(self, X_train, y_train, scenario_name, path):
        super().__init__(X_train, y_train, scenario_name, path)

    def _objective(self, trial):
        # Seleciona o kernel primeiro para poder condicionar outros parâmetros
        kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
    
        params = {
            'C': trial.suggest_float('C', 0.1, 100.0, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': kernel,
            'tol': trial.suggest_float('tol', 1e-5, 1e-1, log=True),
            'shrinking': trial.suggest_categorical('shrinking', [True, False]),
            'random_state': 14
        }
    
        # Hiperparâmetros condicionais
        if kernel == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 4)
            params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)
        elif kernel == 'sigmoid':
            params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)
    
        clf = SVC(**params, probability=True)
    
        # Cross-validation
        score = cross_val_score(clf, self.X_train, self.y_train.values.ravel(), scoring='accuracy', cv=3, n_jobs=-1).mean()
        return score

    def _build_model(self, params):
        return SVC(**params, probability=True, random_state=14)
        
