# Nanodegree Engenheiro de Machine Learning
## Proposta de projeto final
Gustavo Campos 
16 de Abril de 2018

## Proposta

### Histórico do assunto

Medir a popularidade de uma notícia, post, tweet, etc é um dos grandes desafios que enfrentamos hoje. Diversos estudos são feitos nesse sentido para tentar entender como é a propagação e a aceitação de notícias na internet.

Entender quais formas ou quais caracteríscas devem ter uma notícia para ser mais popular pode empoderar os jornais, portais e redes sociais para terem um alcance cada vez maior.

Vou usar esse trabalho como o start para eu começar a estudar a popularidade de forma geral, de vídeos, notícias, notícias fakes e etc.Para tentar entender o que faz por exemplo um vídeo, um texto, um post, etc, com um objetivo de conversão converter ou não.

### Descrição do problema

o projeto é baseado em dados coletados do site de notícias [Mashable](https://mashable.com/) e o dataset resume um conjunto heterogêneo de features sobre noticias publicadas por um período de dois anos. O objetivo principal desse projeto é prever a quantidade de compartilhamentos **(popularidade)** das notícias.

### Conjuntos de dados e entradas

Esse projeto está disponível no site do [Kaggle](https://www.kaggle.com/c/predicting-online-news-popularity) em forma de competição. Alêm disso foi disponibilizado o data set pelo site da [UCI(Repositório de dataset para machine learning)](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity).


Informações sobre Atributos:

Número de Atributos: 61 (58 atributos preditivos, 2 não preditivos, 1 campo de meta) 

Informação do Atributo: 

0. url: URL do artigo (não preditivo) 
1. timedelta: Dias entre a publicação do artigo e a aquisição do conjunto de dados (não -preditivo) 
2. n_tokens_title: Número de palavras no título 
3. n_tokens_content: Número de palavras no conteúdo 
4. n_unique_tokens: Taxa de palavras únicas no conteúdo 
5. n_non_stop_words: Taxa de palavras sem parar no conteúdo 
6. n_non_stop_unique_tokens : Taxa de palavras únicas que não param no conteúdo 
7. num_hrefs: Número de links 
8. num_self_hrefs: Número de links para outros artigos publicados pelo Mashable 
9. num_imgs: Número de imagens 
10. num_videos: Número de vídeos 
11. average_token_length: comprimento médio das palavras no conteúdo 
12. num_keywords: Número de palavras-chave nos metadados 
13. data_channel_is_lifestyle: É o canal de dados 'Lifestyle'? 
14. data_channel_is_entertainment: é o canal de dados 'Entertainment'? 
15. data_channel_is_bus: é o canal de dados 'Business'? 
16. data_channel_is_socmed: é o canal de dados 'Social Media'? 
17. data_channel_is_tech: é o canal de dados 'Tech'? 
18. data_channel_is_world: é o canal de dados 'World'? 
19. kw_min_min: Pior palavra-chave (min. Compartilhamentos) 
20. kw_max_min: Pior palavra-chave (máx. Compartilhamentos) 
21. kw_avg_min: Pior palavra-chave (média de compartilhamentos) 
22. kw_min_max: Melhor palavra-chave (min. Compartilhamentos) 
23. kw_max_max: Melhor palavra-chave (máximo de compartilhamentos) 
24. kw_avg_max: Melhor palavra-chave (média de compartilhamentos) 
25. kw_min_avg: Média. palavra-chave (min. compartilhamentos) 
26. kw_max_avg: Avg. palavra-chave (máx. compartilhamentos) 
27. kw_avg_avg: média palavra-chave (média de compartilhamentos) 
28. self_reference_min_shares: min. compartilhamentos de artigos referenciados no Mashable 
29. self_reference_max_shares: Max. compartilhamentos de artigos referenciados no Mashable 
30. self_reference_avg_sharess: Avg. ações de artigos referenciados no Mashable 
31. weekday_is_monday: O artigo foi publicado na segunda-feira? 
32. weekday_is_tuesday: O artigo foi publicado em uma terça-feira? 
33. weekday_is_wednesday: O artigo foi publicado em uma quarta-feira? 
34. weekday_is_thursday: O artigo foi publicado em uma quinta-feira? 
35. weekday_is_friday: O artigo foi publicado em uma sexta-feira? 
36. weekday_is_saturday: O artigo foi publicado em um sábado? 
37. weekday_is_sunday: O artigo foi publicado em um domingo? 
38. is_weekend: O artigo foi publicado no final de semana? 
39. LDA_00: Proximidade do tópico LDA 0 
40. LDA_01: Proximidade do tópico 1 do LDA 
41. LDA_02: Proximidade do tópico 2 do LDA 
42. LDA_03: Proximidade do tópico 3 do LDA 
43. LDA_04: Proximidade do tópico 4 do LDA 
44. global_subjectivity: Text subjetividade 
45. global_sentiment_polarity: polaridade do sentimento do texto 
46. ​​global_rate_positive_words: Taxa de palavras positivas no conteúdo 
47. global_rate_negative_words: Taxa de palavras negativas no conteúdo 
48. rate_positive_words: Taxa de palavras positivas entre tokens não neutros 
49. rate_negative_words: Taxa de palavras negativas entre tokens não neutros 
50. avg_positive_polarity: média polaridade de palavras positivas 
51. min_positive_polarity: min. polaridade das palavras positivas 
52. max_positive_polarity: max. polaridade das palavras positivas 
53. avg_negative_polarity: média polaridade das palavras negativas 
54. min_negative_polarity: min. polaridade de palavras negativas 
55. max_negative_polarity: máx. polaridade das palavras negativas 
56. title_subjectivity: Subject subjectivity 
57. title_sentiment_polarity: polaridade do título 
58. abs_title_subjectivity: nível de subjetividade absoluta 
59. abs_title_sentiment_polarity: nível de polaridade absoluta 
60. shares: número de compartilhamentos (target)


Algumas observações sobre o dataset retirado do site da UCI:

- Os artigos foram publicados pela Mashable (www.mashable.com) e seu conteúdo como direitos de reprodução pertence a eles. Portanto, esse conjunto de dados não compartilha o conteúdo original, mas algumas estatísticas associadas a ele. O conteúdo original pode ser acessado e recuperado publicamente usando os URLs fornecidos. 
- Data de aquisição: 8 de janeiro de 2015 
- Os valores estimados de desempenho relativo foram estimados pelos autores usando um classificador Random Forest e uma janela de rolagem como método de avaliação. [Veja o artigo deles](https://www.researchgate.net/publication/283510525_A_Proactive_Intelligent_Decision_Support_System_for_Predicting_the_Popularity_of_Online_News) para mais detalhes sobre como os valores relativos de desempenho foram definidos.

Algun desafios que são inerentes a medidas de popularidade e entendimento de aceitação/propagação de noticias tais como análise de sentimento e palavras não fazem parte deste trabalho, somente serão utilizados as features que foram fornecidas pelo dataset.

### Descrição da solução

O objetivo desse projeto é criar um modelo de classificação para predizer a popularidade de uma notícia, para esse experimento utilizaremos os dados do dataset para criar um classificador e após a classificação tentar predizer qual seria a popularidade da notícias (quantidade de compartilhamentos).


### Modelo de referência (benchmark)

A idéia de ter um modelo treinado utilizando o dataset é fornecer um classficador com capacidade de predizer a popularidade de uma nova noticia com uma boa precisão, sendo assim é razoável dizer que o modelo treinado pelo dataset teria uma melhor performance que uma geração de compartilhamento(popularidade) aleatório.

Uma outra forma de ter um modelo de referência para validar o modelo treinado é treinar uma regressão para a quantidade de compartilhamentos com o dataset não trabalhado, após treinado retirar uma métrica de R2 para medir o quanto o modelo treinado consegue se ajustar a amostra.

### Métricas de avaliação

Para verificar o desempenho da regressão utilizarei  métrica de R2, que siginifica a variância do modelo, em outras palavras o quanto o modelo treinado no dados de treino consegue ser ajustado nos dados de teste.

Para verificar os classificadores vou utilizar as métricas da Matriz de Confusão: 

Accuracy - A proporção de predições corretas, sem levar em consideração o que é positivo e o que é negativo.
Precision - A proporção de verdadeiros positivos em relação a todas as predições positivas.
Recall - A capacidade do sistema em predizer corretamente a condição para casos que realmente a têm
F1-Score - Essa métrica é uma combinação da precisão recall, traz um número único que indique a qualidade geral do seu modelo.

### Design do projeto

Na primeira etapa do projeto tenho como objetivo explorar os dados e as features do dataset:

Verificar e tentar enteder a distribuição dos dados pelas features que possuem dados numéricos, e da mesma forma das features que possuem dados categóricos/discretos.

Tentar entender a interação entre as diversas categorias/features presentes no dataset, entender o corelacionamento e caso seja possível extrair as componentes principais do dataset.

Entender quais a features mais relavantas e retirar features que não são relevantes para o rótulo, para evitar o overfeating.

Em uma segunda etapa a idéia é tentar entender se existem possíveis segmentações entre a distribuição de popularidade:

Crias novas colunas com um level(baixo / medio / alto) de popularidade, e uma nova coluna com binária dizendo se é popular ou não de acordo com a média de compartilhamentos.

Em uma próxima etapa vou separar os dados em dados de treinamento e dados de teste, utilizando a biblioteca do python sklearn.cross_validation.

Em uma quarta etapa vou criar classificadores e treina-los para classificar de acordo com essas novas colunas (é_popular, level_popularidade), os algoritimos utilizados nesse treinamento serão  SVM, Arvore de decisão, Regressão logistica, Naive Bayses e Ensemble Methods.

Por útimo vou calibrar o modelo escolhido utilizando a bliboteca sklearn as tecnicas grid_search.gridSearchCV metrics.make_scorer, para tunar os classificadores.

Enfim, verificar as métricas escolhidas e o quanto o modelo é capaz de generalizar.

### Referências

K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.

https://mineracaodedados.wordpress.com/tag/matriz-de-confusao/



