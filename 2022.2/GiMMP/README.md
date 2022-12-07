



# `Dê-me-um-filme-por-favor`
# `GiMMP (Give-Me-a-Movie-Please)`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA930 - Computação Afetiva*,  oferecida no segundo semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

 |Nome  | RA | Especialização|
 |--|--|--|
 | Guilherme Camargo  | 201664  | Eng. de Computação|


## Descrição do Projeto

Nos últimos anos, a internet vem sendo utilizada por grande parte das pessoas para entretenimento, e assistir filme(s) neste contexto torna-se algo comum. Contudo, escolher um título do imenso catálogo disponível nem sempre é tarefa fácil. A fim de mitigar tal problema, tem-se como objetivo deste projeto o desenvolvimento de um software capaz de sugerir filme(s) de acordo com a emoção detectada, através de imagem(ns) facial(is), do usuário [1, 2], ou seja, implementar um sistema afetivo de recomendação de filmes.

Portanto, o projeto será constituído de dois módulos principais:

 1. **Construção do modelo de classificação**: Construir um modelo de Inteligência Artificial capaz de detectar emoções em imagens de faces humanas.
 
 3. **Sugerir filme**: Mapear o resultado obtido através do modelo com um ou mais títulos do catálogo de filmes.

Vídeo: https://youtu.be/KHQICMc6GZo


## Abordagem Adotada

### Conjuntos de dados
Dois conjuntos de dados foram utilizados neste projeto:

 1. **Extended Cohn-Kanade dataset (CK+)**: Base de dados com aproximadamente 981 imagens de faces humanas, subdivididas em 7 expressões faciais (0=Anger, 1=Contempt, 2=Disgust, 3=Fear, 4=Happy, 5=Sadness, 6=Surprise).  O conjunto de dados CK+ foi dividido da seguinte maneira:
 
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
 
 2. **TMDB 5000 Movie Dataset** [7]: Base de dados com informações de filmes de diversos gêneros (Drama, Comedy, Thriller, Action, Romance, Adventure, Crime, Science Fiction, Horror, Family, Fantasy, Mystery, Animation, History, Music, War, Documentary, Western, Foreign, TV Movie).

A princípio o conjunto de expressões faciais FER-2013 [6] seria utilizado. Contudo foi substituído por um conjunto modificado do CK+ [12] (apenas alguns frames dos vídeos originais foram coletados neste conjunto). O motivo da substituição foi a dificuldade em construir um modelo que generalizasse minimamente bem no conjunto FER-2013. Os resultados das tentativas frustradas estão presentes a seguir:

![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/graficos/dropout%200.25/acc.png?raw=true)
![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/graficos/dropout%200.25/loss.png?raw=true)


### Modelagem adotada
Optou-se por desenvolver o projeto em linguagem de programação Python [3], e a abordagem em Deep Learning foi escolhida pela facilidade (bibliotecas eficientes disponíveis, tais como pytorch [4] e tensorflow [5]) e oportunidade estudar e por em prática o assunto "Deep Learning" em um projeto prático.
Algumas arquiteturas foram testadas ao decorrer do desenvolvimento do projeto e a que obteve melhores resultados em melhores tempos (tradeoff positivo) foi a arquitetura descrita a seguir:

![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/imagens/summary.png?raw=true)

### Mapeamento "emoção vs gênero"

Houve grande dificuldade em encontrar artigos científicos que que mapeassem uma determinada emoção com um determinado gênero de filme. Há grande disponibilidade de trabalhos que respondem **"qual emoção é expressa por um humano ao assistir determinado filme"**, por exemplo, mas nenhum trabalho responde a **"qual tipo [gênero] de filme devo assistir quando estou triste [, bravo, surpreso, com medo, etc]"**.
Por isso, uma busca foi feita na internet buscando artigos não científicos, onde os autores sugeriam filmes ou gêneros de filmes para um determinado "mood" [8, 9, 10, 11].

![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/imagens/imdb.png?raw=true)

Com as listas encontradas nas referências foi possível generalizar os gêneros de filmes para uma determinada emoção. O mapeamento final utilizado é o seguinte:

![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/imagens/mapeamento_emo_genre.png?raw=true)

Com certeza o mapeamento não está perfeito e 100% correto e ainda pode ser melhorado.

### Resultados
A seguir serão apresentados os resultados obtidos deste projeto.

Uma interface gráfica simples foi desenvolvida para melhor apresentação. Algumas screenshots estão disponíveis a seguir:

* Tela inicial do programa.
![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/imagens/gui1.png?raw=true)

* Telas mostrando o rótulo original e rótulo previsto para as emoções (a) Sad, (b) Disgust e (c) Happy. Em cada tela é possível observar a lista (de 10 filmes cada) de filmes sugeridos.
![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/imagens/gui2.png?raw=true)

Abaixo temos os resultados relacionados ao treinamento da rede neural. Como pode ser observado, a acurácia alcançada foi alta (100% por volta da época 150) tanto para o conjunto de treinamento quanto para o conjunto de validação. Isso mostra que o modelo conseguiu generalizar muito bem o problema em questão.

![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/graficos/ck+%20dropout%200.6%20small%20model%20200%20epochs%20lr%201e-4/Figure_2.png?raw=true)

O gráfico de erro mostra que o modelo treinado **não** apresenta nem overfitting (quando o modelo aprende demais/somente sobre os dados de treinamento) ou underfitting (erro elevado tento nos dados de treino quanto nos dados de validação).

![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/graficos/ck+%20dropout%200.6%20small%20model%20200%20epochs%20lr%201e-4/Figure_1.png?raw=true)

A curva ROC abaixo também nos leva a concluir que o classificador gerado possui ótimo desempenho, uma vez linhas acima da pontilhada indicam um melhor classificador (em torno da linha pontilhada significa que o classificador é aleatório; abaixo significa que o classificador é ruim).

![enter image description here](https://github.com/btguilherme/ia930/blob/main/2022.2/GiMMP/graficos/ck+%20dropout%200.6%20small%20model%20200%20epochs%20lr%201e-4/ROC.png?raw=true)

## Discussão/Conclusões
- Conjunto CK+ é muito mais simples que FER-2013 (desafio).
- Tempo de aprendizado do modelo é muito mais rápido e com melhores resultados para o conjunto CK+.
- Utilização de redes pré-treinadas, p.e. ResNet pode ser uma alternativa.
- Mapeamento "emoções vs gênero" parece ser um campo de pesquisa em aberto.
- Utilizar mais parâmetros para sugerir filmes (sexo, idade, nacionalidade, etc.)

## Apresentação final [de slides]
https://docs.google.com/presentation/d/158R6Cg84hSFUIDmqNpi3KVTlbOrMy3Y--fU-KZN6Uss/edit?usp=sharing


## Referências Bibliográficas

[1] Q. Chen and J. Qin, "Research and implementation of movie recommendation system based on deep learning," 2021 IEEE International Conference on Computer Science, Electronic Information Engineering and Intelligent Control Technology (CEI), 2021, pp. 225-228, doi: 10.1109/CEI52496.2021.9574461.URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9574461&isnumber=9574447

[2] K. Amulya, S. B. Swathi, P. Kamakshi and Y. Bhavani, "Sentiment Analysis on IMDB Movie Reviews using Machine Learning and Deep Learning Algorithms," 2022 4th International Conference on Smart Systems and Inventive Technology (ICSSIT), 2022, pp. 814-819, doi: 10.1109/ICSSIT53264.2022.9716550.URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9716550&isnumber=9716227

[3] https://docs.python.org/3/
 
[4] https://pytorch.org/tutorials/
 
[5] https://www.tensorflow.org/tutorials?hl=pt-br
 
[6] https://paperswithcode.com/dataset/fer2013
 
[7] https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

[8] https://psychcentral.com/depression/movies-to-uplift-you-from-depression

[9] https://www.imdb.com/list/ls053456706/

[10] https://www.backtothemovies.com/heres-the-absolute-best-movies-to-watch-when-sick/#:~:text=Here%E2%80%99s%20The%20Absolute%20Best%20Movies%20To%20Watch%20When,Potter%208%20Back%20to%20the%20Future%20Mais%20itens

[11] https://mood2movie.com/

[12] P. Lucey, J. F. Cohn, T. Kanade, J. Saragih, Z. Ambadar and I. Matthews, "The Extended Cohn-Kanade Dataset (CK+): A complete dataset for action unit and emotion-specified expression," 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition - Workshops, 2010, pp. 94-101, doi: 10.1109/CVPRW.2010.5543262.

