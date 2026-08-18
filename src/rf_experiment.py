from src.base_experiment import BaseExperiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class RandomForestExperiment(BaseExperiment):
    # Contrutor da Classe
    def __init__(self, X_train, y_train, scenario_name, path):
        super().__init__(X_train, y_train, scenario_name, path)

    def _objective(self, trial):
        # Hiperparâmetros
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }
    
        clf = RandomForestClassifier(**params, random_state=14)
    
        score = cross_val_score(clf, self.X_train, self.y_train.values.ravel(), scoring='accuracy', n_jobs=-1, cv=3).mean()
        return score

    def _build_model(self, params):
        return RandomForestClassifier(**params, n_jobs=-1, random_state=14)
        
