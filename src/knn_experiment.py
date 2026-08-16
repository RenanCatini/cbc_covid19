from base_experiment import BaseExperiment
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

class KNNExperiment(BaseExperiment):
    # Contrutor da Classe
    def __init__(self, X_train, X_test, y_train, y_test, scenario_name, path):
        super().__init__(X_train, X_test, y_train, y_test, scenario_name, path)

    def _objective(self, trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 30),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 10)
        }
    
        clf = KNeighborsClassifier(**params, n_jobs=-1)
    
        # Validação cruzada com Acurácia para avaliar o desempenho
        score = cross_val_score(clf, self.X_train, self.y_train.values.ravel(), scoring='accuracy', cv=5, n_jobs=-1).mean()
        return score

    def _build_model(self, params):
        return KNeighborsClassifier(**params, n_jobs=-1)
        
