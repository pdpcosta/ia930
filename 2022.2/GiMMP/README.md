
# `Dê-me-um-filme-por-favor`
# `GiMMp (Give-Me-a-Movie-Please)`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA930 - Computação Afetiva*,  oferecida no segundo semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

 |Nome  | RA | Especialização|
 |--|--|--|
 | Guilherme Camargo  | 201664  | Eng. de Computação|


## Descrição Resumida do Projeto

Nos últimos anos, a internet vem sendo utilizada por grande parte das pessoas para entretenimento, e assistir filme(s) neste contexto torna-se algo comum. Contudo, escolher um título do imenso catálogo disponível nem sempre é tarefa fácil. A fim de mitigar tal problema, tem-se como objetivo deste projeto o desenvolvimento de um software capaz de sugerir filme(s) de acordo com a emoção detectada, através de imagem(ns) facial(is), do usuário, ou seja, implementar um sistema afetivo de recomendação de filmes.

Portanto, o projeto será constituído de dois módulos principais:

 1. **Construção do modelo de classificação**: Construir um modelo de Inteligência Artificial capaz de detectar emoções em imagens de faces humanas.
 2. **Sugerir filme**: Mapear o resultado obtido através do modelo com um ou mais títulos do catálogo de filmes.

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

## Cronograma
| Item | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8
|--|--|--|--|--|--|--|--|--|--|--|--|
| Proposta | X | X |  |  |  |  |  |  
| Levantamento dos dados | X | X |  |  |  |  |  | 
| Desenvolvimento do algoritmo |  | X | X | X | X |  |  | 
| Testes |  |  | X | X | X | X |  |  
| Entrega |  |  |  |  |  |  | X |  X

## Referências Bibliográficas
 https://docs.python.org/3/
 https://pytorch.org/tutorials/
 https://www.tensorflow.org/tutorials?hl=pt-br
 https://builtin.com/data-science/supervised-learning-python
 https://paperswithcode.com/dataset/fer2013
 https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
