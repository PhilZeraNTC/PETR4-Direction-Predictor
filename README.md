# 📈 Stock Movement Predictor: PETR4

## 🎯 Sobre o Projeto
Este projeto aplica técnicas de **Machine Learning** para prever a direção do fechamento diário das ações da Petrobras (PETR4.SA). O objetivo é classificar se o preço de fechamento do dia seguinte será maior (1) ou menor/igual (0) ao do dia atual, servindo como base para estratégias de trading quantitativo.

O modelo central utiliza o algoritmo **LightGBM**, e a otimização de hiperparâmetros foi feita de forma automatizada e bayesiana utilizando a biblioteca **Optuna**.

## ⚙️ Engenharia de Features
Os dados foram extraídos via `yfinance` englobando o período de 2015 a 2024. As seguintes variáveis (features) foram criadas para alimentar o modelo:
* **Retorno Diário:** Percentual de mudança de preço dia a dia.
* **Retornos Defasados (Lags 1 e 2):** Capturam o momento de curto prazo do ativo.
* **Preço / SMA_10:** Razão entre o preço de fechamento atual e a Média Móvel Simples de 10 dias.
* **Volatilidade:** Desvio padrão móvel dos retornos nos últimos 10 dias.

## 🛠️ Tecnologias Utilizadas
* **Python**
* **LightGBM:** Treinamento do modelo de Gradient Boosting.
* **Optuna:** Otimização de hiperparâmetros com `LightGBMPruningCallback` para podar tentativas pouco promissoras (Early Stopping).
* **yfinance:** Coleta do histórico de preços.
* **Pandas & NumPy:** Manipulação de dados e cálculo de indicadores financeiros.
* **Scikit-Learn:** Avaliação de métricas do modelo (Accuracy).

## 📊 Resultados Alcançados
O Optuna realizou 50 iterações (trials) para encontrar a melhor combinação de hiperparâmetros focando em maximizar a métrica AUC (Area Under the Curve) e a Acurácia.

* **Melhor Acurácia no Teste (Out-of-sample):** 0.592

  
<img width="1726" height="470" alt="image" src="https://github.com/user-attachments/assets/9c958f9d-85c0-4229-abb5-74d22dd878d3" />


<img width="1716" height="475" alt="image" src="https://github.com/user-attachments/assets/3b253b12-2b8c-4b4c-859b-4ec2633eab44" />




## 🚀 Como Executar
1. Clone o repositório.
2. Instale as dependências:
   ```bash
   pip install optuna optuna-integration lightgbm yfinance pandas numpy scikit-learn
