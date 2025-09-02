# Análise de Regressão no Dataset de Corrupção 

Este documento descreve **exatamente o que foi feito** no notebook `corrupcao_from_Priek.ipynb`, que replica a ordem e a lógica do *PrieKaue (1).ipynb* aplicadas ao arquivo **`corrupcao.csv`**.

---

## 1) Contexto e objetivo
Modelar o **`Corrupcao_Valor_Perdido`** a partir de variáveis econômicas e institucionais usando:
- **Regressão Linear (OLS)** como baseline;
- **Regressão Polinomial (grau 2)**;
- **Seleção de variáveis via Backward Elimination** (α = 0,05).

---

## 2) Visão do dataset
- **Tamanho:** 100 linhas × 11 colunas
- **Variáveis numéricas (exemplos):** `Orcamento_Total`, `Salario_Medio_Func`, `Numero_Fiscalizacoes`, `Contratos_Publicos`, `Indice_Transparencia`, `Corrupcao_Valor_Perdido` (alvo)
- **Variáveis categóricas (exemplos):** `Partido_Governo`, `Midia_Livre`, `Judiciario_Atuante`, `Cultura_Corrupcao`, `Investigacoes_Federais`

---

## 3) O que foi feito (passo a passo)

### 3.1 Diagnóstico inicial
- `df.info()` e `df.head()`
- **Contagem de valores únicos** por coluna
- **Mapa de nulos** (visual)
- **Histogramas** das variáveis numéricas

### 3.2 Tratamento de dados
- Valores **negativos** nas colunas numéricas relevantes → convertidos para `NaN`
- **Imputação numérica:** mediana
- **Imputação categórica:** moda
- Checagem final de nulos e **estatísticas pós-tratamento**

### 3.3 Preparação para modelagem
- Conversão de categóricas em **dummies** (`drop_first=True`)
- Garantia de tipos **numéricos**
- Definição de `X` e `y` com `y = Corrupcao_Valor_Perdido`
- **Split** treino/teste em 70%/30% (random_state=42)

### 3.4 Regressão Linear (OLS) com diagnósticos
- Ajuste por `statsmodels.OLS`
- Cálculo de **R² (treino/teste)**
- Testes nos resíduos:
  - **Breusch–Pagan** (heterocedasticidade)
  - **Durbin–Watson** (autocorrelação)
  - **Shapiro–Wilk** (normalidade)
- `summary()` completo do modelo

### 3.5 Regressão Polinomial (grau 2)
- Geração de *features* com `PolynomialFeatures(degree=2, include_bias=False)`
- Novo ajuste OLS e **mesmos diagnósticos** (R², BP, DW, Shapiro)
- `summary()` do modelo polinomial

### 3.6 Seleção de variáveis (*Backward Elimination*)
- Remoção iterativa de preditores com p-valor > 0,05 no **treino**
- `summary()` do **modelo final**
- Versão **polinomial** restrita às *features* selecionadas e respectivos testes

### 3.7 Gráficos finais (no notebook)
- **Valores Reais vs. Preditos** (teste)
- **Resíduos vs. Preditos** (teste)

---

## 4) Principais resultados do run atual

### 4.1 OLS (baseline)
- **R² treino:** 0,5925  
- **R² teste:** −0,2808  → *não generaliza* (overfitting / baixo sinal)
- **Breusch–Pagan (p):** 0,8908 → **homocedástico**
- **Durbin–Watson:** 1,71 → leve **autocorrelação positiva**
- **Shapiro–Wilk (p):** 0,9572 → resíduos **compatíveis com normalidade**

### 4.2 Backward Elimination (treino)
- **Qtd. de *features* selecionadas:** 1  
- **Variável mantida:** `Orcamento_Total`

### 4.3 Polinomial (grau 2) com features selecionadas
- **R² treino:** 0,5293  
- **R² teste:** 0,1535  → pequena melhoria vs. baseline
- **Breusch–Pagan (p):** 0,3353 → **homocedástico**
- **Durbin–Watson:** 1,67 → leve **autocorrelação positiva**
- **Shapiro–Wilk (p):** 0,4347 → resíduos **compatíveis com normalidade**

> **Leitura geral:** O conjunto de dados é pequeno e o **sinal preditivo parece concentrado em `Orcamento_Total`**; os demais preditores (incluindo dummies) foram descartados no *backward*. O modelo polinomial com a *feature* selecionada melhora a generalização (R²_test ≈ 0,15), mas ainda indica **baixa capacidade explicativa** — sugerindo necessidade de novas variáveis, transformação/regularização e/ou aumento de amostra.

---

## 5) Artefatos gerados no notebook
- Tabelas de diagnóstico (`info`, `describe`, contagem de únicos)
- Mapa de nulos e histogramas (exibidos inline)
- **Resumos OLS** (`model.summary()`) para os modelos linear e polinomial
- Gráficos: **Reais vs. Preditos** e **Resíduos vs. Preditos** (exibidos inline)

---

## Decisões e recomendações (o que dá pra fazer com isso)

**1) Priorizar fiscalização por orçamento (regra simples e prática).**  
O *backward elimination* manteve apenas **`Orcamento_Total`** como preditor relevante do **`Corrupcao_Valor_Perdido`**. A associação é **positiva e estatisticamente significativa** (p ≈ 1.1e-12). No ajuste linear, o coeficiente foi ≈ **0,0518**: em média, **cada R$ 1 milhão de orçamento está associado a ~R$ 51,8 mil a mais** de perda estimada por corrupção (associação, não causalidade).  
👉 **Decisão prática:** defina faixas de risco por **quartis de orçamento** e **enderece sua capacidade de auditoria** começando pelo topo:

- Q1 (≈ R$ 44,0 mi): perda estimada ≈ **R$ 2,06 mi**  
- Q2 (≈ R$ 50,9 mi): perda estimada ≈ **R$ 2,42 mi**  
- Q3 (≈ R$ 56,6 mi): perda estimada ≈ **R$ 2,72 mi**  
- Máx. observado (R$ 200 mi): perda estimada ≈ **R$ 10,15 mi**

> Exemplo de governança: municípios/órgãos **no Q3+** entram em **prioridade alta** com mais amostragens em contratos, *follow-ups* obrigatórios e trilhas de auditoria mais rígidas.

**2) Usar a linha de tendência como “meta de referência”, não como verdade absoluta.**  
A regressão linear sozinha **não generalizou bem** (R²_test < 0). Já a versão **polinomial com features selecionadas** melhora para **R²_test ≈ 0,15** — ainda baixo.  
👉 **Decisão prática:** trate o valor previsto como **faixa de referência** para metas de redução. Se um órgão gastar bem acima do previsto para seu orçamento, isso **acende alerta** e justifica auditoria direcionada.

**3) Regras de triagem para alocação de esforço.**
- **Regra de capacidade:** planejar **nº de fiscalizações proporcional ao orçamento** (ex.: pontos de auditoria por cada R$ 10 mi).  
- **Regra de alerta:** se o **valor perdido observado** ficar **>30–50%** acima da faixa prevista para aquele orçamento, abrir **investigação adicional**.  
- **Regra de acompanhamento:** órgãos que migram de Q2→Q3 no orçamento entram automaticamente em **monitoramento reforçado** no próximo ciclo.

**4) O que *não* dá pra concluir (e por quê).**  
Variáveis institucionais/categóricas (ex.: `Partido_Governo`, `Judiciario_Atuante`, `Midia_Livre`, etc.) **não ficaram no modelo final** nesta amostra. Isso **não comprova** irrelevância; pode ser **amostra pequena**, colinearidade com orçamento ou codificação simplificada.  
👉 **Decisão prática:** **não basear políticas** apenas nessas categorias **sem mais evidência**. Se forem estratégicas, coletar dados mais granulares (ex.: tipo de licitação, concentração de fornecedores, aditivos, histórico de sanções, *red flags* de compras).

**5) Melhorias de dado/modelo que destravam decisões mais fortes.**
- **Granularidade de gasto:** separar orçamento/perdas por **categoria de despesa/contrato** (obras, saúde, educação…) para ação setorial.  
- **Histórico temporal:** séries mensais/trimestrais para tratar a **autocorrelação** (DW ~ 1,7); modelos de painel/tempo tendem a render insights melhores.  
- **Indicadores de integridade:** incluir **red flags** (dispensa recorrente, fornecedor único, aditivos altos, fatiamento) — diretamente acionáveis.  
- **Regularização e validação:** Ridge/Lasso + *cross-validation* para reduzir sobreajuste e selecionar variáveis de forma mais robusta.

---

## 6) Observações finais
- Há sinais de **multicolinearidade** em ajustes com muitas *dummies* (condition number alto nos `summary()`).
- A **autocorrelação leve** nos resíduos (DW ~1,7) merece atenção para modelos temporais/espaciais.
- **Próximos passos sugeridos:** novas fontes de dados, *feature engineering* dirigido e avaliação com regularização (Ridge/Lasso) e validação cruzada.
