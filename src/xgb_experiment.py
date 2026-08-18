from src.base_experiment import BaseExperiment
import xgboost as xgb
from sklearn.model_selection import cross_val_score

class XGBExperiment(BaseExperiment):
    # Contrutor da Classe
    def __init__(self, X_train, y_train, scenario_name, path):
        super().__init__(X_train, y_train, scenario_name, path)

    def _objective(self, trial):
        # Hiperparâmetros recomendados para XGBoost
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_jobs': -1,
            'random_state': 14
        }
    
        # Inicializa o classificador
        clf = xgb.XGBClassifier(**params)
    
        # Cross-validation
        score = cross_val_score(clf, self.X_train, self.y_train.values.ravel(), scoring='accuracy', cv=3).mean()
        return score

    def _build_model(self, params):
        return xgb.XGBClassifier(**params, objective='binary:logistic', n_jobs=-1, random_state=14)
        
