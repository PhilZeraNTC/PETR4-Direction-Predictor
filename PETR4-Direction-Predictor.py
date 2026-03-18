
!pip install optuna optuna-integration lightgbm yfinance

import optuna
import lightgbm as lgb
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# coleta de dados
df = yf.download('PETR4.SA', start='2015-01-01', end='2024-01-01')
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

col_price = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

# engenharia de features (inicial)
df['Retorno'] = df[col_price].pct_change()
df['Target'] = np.where(df[col_price].shift(-1) > df[col_price], 1, 0)
df['Retorno_Lag1'] = df['Retorno'].shift(1)
df['Retorno_Lag2'] = df['Retorno'].shift(2)
df['SMA_10'] = df[col_price].rolling(10).mean()
df['Preco_SMA'] = df[col_price] / df['SMA_10']
df['Volatilidade'] = df['Retorno'].rolling(10).std()
df.dropna(inplace=True)

# separando treino de teste
train = df[df.index < '2023-01-01']
test = df[df.index >= '2023-01-01']

features = ['Retorno', 'Retorno_Lag1', 'Retorno_Lag2', 'Preco_SMA', 'Volatilidade']
X_train, y_train = train[features], train['Target']
X_test, y_test = test[features], test['Target']

# criando o dataset lightgbm
dtrain = lgb.Dataset(X_train, label=y_train)
dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)







def objective(trial):
    # parametros (melhorar)
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': 42,

        # --- A CORREÇÃO ESTÁ AQUI ---
        'feature_pre_filter': False,  # Desativa o filtro prévio para permitir mudar min_data_in_leaf

        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True)
    }

    # podando callbacks
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'auc')

    # treinando o modelo
    gbm = lgb.train(
        param,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(stopping_rounds=20), pruning_callback]
    )

    # avaliando 
    preds = gbm.predict(X_test)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_test, pred_labels)

    return accuracy





# criando study
study = optuna.create_study(direction='maximize', study_name="LGBM_Finance")

print("Iniciando otimização...")
study.optimize(objective, n_trials=50)











best_params = study.best_params

# parametros fixos
best_params['objective'] = 'binary'
best_params['metric'] = 'auc'
best_params['verbose'] = -1

# treinando modelo final com força total
final_model = lgb.train(
    best_params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtest],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

print("Modelo Final treinado")



print("-" * 50)
print("MELHORES RESULTADOS:")
print(f"Melhor Acurácia: {study.best_value:.4f}")
print("Melhores Parâmetros:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")



from optuna.visualization import plot_optimization_history, plot_param_importances

# evoluçao da acuracia
plot_optimization_history(study).show()

# melhores hiperparametros
plot_param_importances(study).show()
