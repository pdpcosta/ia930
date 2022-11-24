


# `Dê-me-um-filme-por-favor`
# `GiMMP (Give-Me-a-Movie-Please)`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA930 - Computação Afetiva*,  oferecida no segundo semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

 |Nome  | RA | Especialização|
 |--|--|--|
 | Guilherme Camargo  | 201664  | Eng. de Computação|


## Descrição Resumida do Projeto

Nos últimos anos, a internet vem sendo utilizada por grande parte das pessoas para entretenimento, e assistir filme(s) neste contexto torna-se algo comum. Contudo, escolher um título do imenso catálogo disponível nem sempre é tarefa fácil. A fim de mitigar tal problema, tem-se como objetivo deste projeto o desenvolvimento de um software capaz de sugerir filme(s) de acordo com a emoção detectada, através de imagem(ns) facial(is), do usuário, ou seja, implementar um sistema afetivo de recomendação de filmes.

Portanto, o projeto será constituído de dois módulos principais:

 1. **Construção do modelo de classificação**: Construir um modelo de Inteligência Artificial capaz de detectar emoções em imagens de faces humanas.
 
 3. **Sugerir filme**: Mapear o resultado obtido através do modelo com um ou mais títulos do catálogo de filmes.

Vídeo: https://youtu.be/KHQICMc6GZo


## Metodologia Proposta

A princípio serão utilizados dois conjuntos de dados:

 1. **Facial Expression Recognition 2013 Dataset (FER-2013)**: Base de dados com aproximadamente 30.000 imagens de faces humanas, subdivididas em 7 expressões faciais (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
 2. **TMDB 5000 Movie Dataset**: Base de dados com informações de filmes de diversos gêneros (Drama, Comedy, Thriller, Action, Romance, Adventure, Crime, Science Fiction, Horror, Family, Fantasy, Mystery, Animation, History, Music, War, Documentary, Western, Foreign, TV Movie).

Duas possíveis abordagens: Deep Learning (DL) ou Supervised Learning (SL).

Artigos de referência:
[1] Q. Chen and J. Qin, "Research and implementation of movie recommendation system based on deep learning," 2021 IEEE International Conference on Computer Science, Electronic Information Engineering and Intelligent Control Technology (CEI), 2021, pp. 225-228, doi: 10.1109/CEI52496.2021.9574461.Abstract: In recent years, with the rapid development of information technology and the Internet, watching movies through the Internet has become a habit for many people. However, the overload of movie information has become more and more serious because people cannot get their favorite movie content quickly from the huge amount of movie resources. As one of the important means to alleviate the information overload problem, the recommendation system can help users find their favorite movie content quickly and bring them a good experience, so it is widely used in famous movie and video websites at home and abroad, and has brought great commercial value. In this paper, we introduce the recommendation system and recommendation algorithm, improve the ConvMF model based on deep learning, verify and analyze the experimental results, and finally complete the design of movie recommendation subsystem.URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9574461&isnumber=9574447

[2] K. Amulya, S. B. Swathi, P. Kamakshi and Y. Bhavani, "Sentiment Analysis on IMDB Movie Reviews using Machine Learning and Deep Learning Algorithms," 2022 4th International Conference on Smart Systems and Inventive Technology (ICSSIT), 2022, pp. 814-819, doi: 10.1109/ICSSIT53264.2022.9716550.Abstract: Sentiment analysis is the study, to classify the text based on customer reviews which can provide valuable information to improve business. Previously the analysis was carried out based on the information provided by the customers using natural language processing and machine learning. In this paper, sentiment analysis on IMDB movie reviews dataset is implemented using Machine Learning (ML) and Deep Learning (DL) approaches to measure the accuracy of the model. ML algorithms are the traditional algorithms that work in a single layer while deep learning algorithms work on multilayers and gives better output. This paper helps the researchers to identify the best algorithm for sentiment analysis. The comparison of the machine learning and deep learning approaches shows that DL algorithms provide accurate and efficient results.URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9716550&isnumber=9716227

Caso a abordagem por DL seja escolhida, serão utilizadas as ferramentas Pytorch e/ou TensorFlow. Caso contrário, classificadores de padrão como SVM, RF, OPF, etc.

Espera-se que o programa seja capaz de identificar a emoção em uma imagem ou foto com alta acurácia (~90%) para que o filme sugerido faça sentido.

## Atualizações

### Conjunto de dados

O conjunto de expressões faciais FER-2013 foi substituído por um conjunto modificado do CK+ (apenas alguns frames dos vídeos originais foram coletados neste conjunto). O motivo da substituição foi a dificuldade em construir um modelo que generalizasse minimamente bem no conjunto FER-2013. Os resultados das tentativas frustradas estão presentes a seguir:

![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/graficos/dropout%200.25/acc.png?raw=true)
![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/graficos/dropout%200.25/loss.png?raw=true)

 1. **Extended Cohn-Kanade dataset (CK+)**: Base de dados com aproximadamente 981 imagens de faces humanas, subdivididas em 7 expressões faciais (0=Anger, 1=Contempt, 2=Disgust, 3=Fear, 4=Happy, 5=Sadness, 6=Surprise).
 
 O conjunto de dados CK+ foi dividido da seguinte maneira:
| Emotion | Total (100%) | Train (80% * Total) | Validation (15% * Total) | Test (5% * Total) | 
|--|--|--|--|--|
| Anger | 135 | 108 | 20 | 7 | 
| Contempt | 54 | 43 | 8 | 3  | 
| Disgust | 177 | 142 | 27 | 9 | 
| Fear | 75 | 60 | 11 | 4 | 
| Happy | 207 | 166 | 31 | 10 | 
| Sadness | 84 | 67 | 13 | 4 | 
| Surprise | 249 | 199 | 37 | 12 |

Onde as colunas informam a emoção, o total de imagens, quantidade de imagens para o treinamento, quantidade de imagens para a validação e a quantidade de imagens para testes, respectivamente.

Após a troca de conjuntos de expressões faciais e alguns ajustes no código fonte, o resultado foi o seguinte:

![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/graficos/ck+%20dropout%200.6%20small%20model%20200%20epochs%20lr%201e-4/Figure_2.png?raw=true)

![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/graficos/ck+%20dropout%200.6%20small%20model%20200%20epochs%20lr%201e-4/Figure_1.png?raw=true)

### Abordagem escolhida

O projeto foi desenvolvido em linguagem de programação Python, e a abordagem em Deep Learning foi escolhida pela facilidade (bibliotecas eficientes disponíveis, tais como pytorch e tensorflow) e oportunidade estudar e por em prática o assunto "Deep Learning" em um projeto prático.
Algumas arquiteturas foram testadas ao decorrer do desenvolvimento do projeto e a que obteve melhores resultados em melhores tempos (tradeoff positivo) foi a arquitetura descrita a seguir:

![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/imagens/summary.png?raw=true)

### Mapeamento "emoção vs gênero"

Houve grande dificuldade em encontrar artigos científicos que que mapeassem uma determinada emoção com um determinado gênero de filme. Há grande disponibilidade de trabalhos que respondem **"qual emoção é expressa por um humano ao assistir determinado filme"**, por exemplo, mas nenhum trabalho responde a **"qual tipo [gênero] de filme devo assistir quando estou triste [, bravo, surpreso, com medo, etc]"**.
Por isso, uma busca foi feita na internet buscando artigos não científicos, onde os autores sugeriam filmes ou gêneros de filmes para um determinado "mood".

![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/imagens/imdb.png?raw=true)

Lista de referências:
[https://psychcentral.com/depression/movies-to-uplift-you-from-depression](https://psychcentral.com/depression/movies-to-uplift-you-from-depression)
https://www.imdb.com/list/ls053456706/
https://www.backtothemovies.com/heres-the-absolute-best-movies-to-watch-when-sick/#:~:text=Here%E2%80%99s%20The%20Absolute%20Best%20Movies%20To%20Watch%20When,Potter%208%20Back%20to%20the%20Future%20Mais%20itens

Com essas listas foi possível generalizar os gêneros de filmes para uma determinada emoção. 

![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/imagens/mapeamento_emo_genre.png?raw=true)

Com certeza o mapeamento não está perfeito e 100% correto e ainda pode ser melhorado.

## Apresentação [de slides] final
https://docs.google.com/presentation/d/158R6Cg84hSFUIDmqNpi3KVTlbOrMy3Y--fU-KZN6Uss/edit?usp=sharing

## Conclusões
- Conjunto CK+ é muito mais simples que FER-2013 (desafio).
- Tempo de aprendizado do modelo é muito mais rápido e com melhores resultados.
- Taxa de aprendizagem é muito importante para o desenvolvimento do modelo.
- Utilização de redes pré-treinadas, p.e. ResNet.
- Mapeamento "emoções vs gênero" parece ser um campo de pesquisa em aberto.


## Cronograma

A seguir o possível cronograma:

| Item | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 |
|--|--|--|--|--|--|--|--|--|
| Proposta | X | X |  |  |  |  |  |  |
| Levantamento dos dados | X | X |  |  |  |  |  |  |
| Desenvolvimento do algoritmo |  | X | X | X | X |  |  |  |
| Testes |  |  | X | X | X | X |  |  |
| Entrega |  |  |  |  |  |  | X | X |


## Referências Bibliográficas
 https://docs.python.org/3/
 
 https://pytorch.org/tutorials/
 
 https://www.tensorflow.org/tutorials?hl=pt-br
 
 https://builtin.com/data-science/supervised-learning-python
 
 https://paperswithcode.com/dataset/fer2013
 
 https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

https://psychcentral.com/depression/movies-to-uplift-you-from-depression

https://www.imdb.com/list/ls053456706/

https://www.backtothemovies.com/heres-the-absolute-best-movies-to-watch-when-sick/#:~:text=Here%E2%80%99s%20The%20Absolute%20Best%20Movies%20To%20Watch%20When,Potter%208%20Back%20to%20the%20Future%20Mais%20itens