# Classificando a qualidade de vinhos com Machine Learning 

## 1. Compreensão do problema

O objetivo do relatório é documentar a construção de uma pipeline de Machine Learning para a **classificação da qualidade de vinhos, com base em suas características físico-químicas**. A base de dados utilizada é a Wine Quality Dataset, [disponível no Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) e o resultado almejado é auxiliar enólogos na tomada de decisão e padronização do processo produtivo, almejando a obtenção de vinhos de alta qualidade. 

Para atingir esse objetivo, foram construídos dois modelos, um baseado no algoritmo **K-Nearest Neighbors (KNN)** e o outro no **Random Forest**. Em ambos os casos, a variável alvo é *high_quality*, que foi criada como uma classificação binária da coluna *quality*: No caso de quality ser ≥ 7 então *high_quality* é igual a 1, senão é igual a 0.

## 2. Análise Exploratória de Dados (EDA)

### 2.1 Identificar e justificar a correlação entre as variáveis

<img width="755" height="392" alt="image" src="https://github.com/user-attachments/assets/59d0032f-0c87-41b1-a472-047bd9123d11" />

* **Alcohol (0.40):** A relação entre *alcohol* é *high_quality* é moderada (+0.4) e diretamente proporcional. Na experiência degustação do vinho, a quantidade de álcool se relaciona com o "corpo" da bebida, sendo assim, vinhos de 12,5% a 14% tende a ter corpo médio, que agrada a maioria dos paladares (EVINO, [s.d.]).

* **Volatile Acidity (-0.30):** A relação entre *Volatile Acidity* e *high_quality* é moderada (-0.3) e inversamente proporcional. A acidez volatil é associada ao sabor de vinagre no vinho, prejuficando ao paladar e acendendo um alerta sobre a deterioração da bebida (WINE ENTHUSIAST, [s.d.]).
  
* **Citric Acidity (0.25):** A relação entre *Citric Acidity* e *high_quality* é fraco (0.25) e diretamente proporcional. O ácido cítrico está presente em baixa quantidade nas uvas, mas em muitos casos ele é adicionado como um conservante para o vinho (BIOTECSUL, [s.d.]).
  
* **Sulphates (0.21):** A relação entre *Sulphates* e *high_quality* é fraca (0.21) e diretamente proporcional. Os sulfitos podem surgir naturalmente durante a fermentação do vinho ou serem adicionados durante o processo com o objetivo de conservar a bebida (RAW WINE, [s.d.]).

Podemos perceber que entre a 4 variáveis que mais estão correlacionadas com *high_quality*, duas delas agem sobre a conservação da bebida (*Citric Acidity* e *Sulphates*) e duas no paladar (*alcohol* e *volatile acidity*). 

Sobre **multicolinearidade**, ou seja, variáveis explicativas que possuem relação entre si, como *Fixed Acidity* e *PH* (-0.69). O entendimento é por manter todas as varíaveis, pois pelo valor da correlação e a escolha dos modelos **K-Nearest Neighbors (KNN)** e **Random Forest** o impacto, caso exista, será mínimo.

### 2.2 Distribuição de variáveis

<img width="1121" height="873" alt="image" src="https://github.com/user-attachments/assets/ebdb50a2-e983-4643-b1c7-07371cbd33a1" />

Em relação a distribuição das variáveis mais correlacionadas com *high_quality* - *alcohol, volatile acidity, citric acidity e sulphates* - todas são **assimetricas para a direita**, mas com variada intensidade. Sobre as demais variáveis qualitativas, a tendência de **assimetria à direita** se mantém. Esse comportamento indica que a massa de dados tende a se concentrar em valores menores, mas que existem outliers formando uma cauda longa.

<img width="352" height="265" alt="image" src="https://github.com/user-attachments/assets/cc3790af-3bfd-4c05-bacc-5662a7db4191" />

Com base no boxplot da variável target *high_quality* e de *alcohol*, é possível perceber que apesar dos vinhos de qualidade (1) possuírem maior mediana no nível de alcool, há um limite entre aumentar o álcool e a qualidade da bebida, visto que existem vinhos de baixa qualidade e altamente alcoólicos como outliers.

### 2.3 Outliers e valores inconsistentes

**Outliers**

<img width="1243" height="621" alt="image" src="https://github.com/user-attachments/assets/99728c2c-c776-48a8-834f-30dd33e1b70b" />

Com base no boxplot das variáveis explicativas do dataset, é possível perceber que a tendência demonstrada no histograma da seção anterior (2.2) é confirmada: **existem outliers**. No entanto, dado o número reduzido de amostras disponíveis, optou-se por manter os outliers e tratá-los conscientemente durante a construção dos modelos

**Valores inconsistentes**

Nesta etapa, foi identificada a **ausência de valores nulos** e a **presença de 125 linhas duplicadas**. Esse valor representa aproximadamente 10% da base de 1.142 registros e a decisão foi por excluir as duplicatas na etapa de pré-processamento, a fim de evitar o overfitting do modelo.

### 2.4 Balanceamento das Classes

## 7. Referências 

1. EVINO. Teor alcoólico do vinho: o que significa ABV? Evino Blog, [s.d.]. Disponível em: https://www.evino.com.br/blog/teor-alcoolico-vinho-abv/. Acesso em: 27 jun. 2026.
2. WINE ENTHUSIAST. What Is Volatile Acidity in Wine? Wine Enthusiast, [s.d.]. Disponível em: https://www.wineenthusiast.com/basics/drinks-terms-defined/volatile-acidity-wine/. Acesso em: 27 jun. 2026.
3. BIOTECSUL. Informativo técnico de produto. Pelotas: Biotecsul, [s.d.]. 1 arquivo PDF. Disponível em: https://www.biotecsul.com.br/userfiles/produtos-arquivos/72e12f4b34c8ed74a74e06ec3f25777f3b5490980131109e304b49409f17d42b.pdf. Acesso em: 27 jun. 2026.
4. RAW WINE. What are sulfites in wine? Raw Wine Learn, [s.d.]. Disponível em: https://www.rawwine.com/learn/what-are-sulfites-in-wine/. Acesso em: 27 jun. 2026.
