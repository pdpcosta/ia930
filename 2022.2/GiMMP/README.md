
# `Dê-me-um-filme-por-favor`
# `GiMMp (Give-Me-a-Movie-Please)`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA930 - Computação Afetiva*,  oferecida no segundo semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

 |Nome  | RA | Especialização|
 |--|--|--|
 | Guilherme Camargo  | 201664  | Eng. de Computação/Bioinformata|


## Descrição Resumida do Projeto
> Descrição do tema do projeto, incluindo contexto gerador, motivação.
> Descrição do objetivo principal do projeto.
> 
> Incluir nessa seção link para vídeo de apresentação da proposta do projeto (máximo 10 minutos).

O objetivo deste projeto é desenvolver um algoritmo capaz de sugerir filme(s) de acordo com a emoção detectada através de imagem(ns).


## Metodologia Proposta
> Para a primeira entrega, a metodologia proposta deve procurar responder:
~~> * Qual(is) base(s) de dado(s) o projeto pretende utilizar, justificando a(s) escolha(s) realizadas.~~
~~> * Quais abordagens o grupo já enxerga como interessantes de serem estudadas.~~
> * Artigos de referência já identificados e que serão estudados ou usados como parte do planejamento do projeto
~~> * Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).~~
~~> * Resultados esperados~~
> * Proposta de avaliação

A princípio serão utilizados dois conjuntos de dados:

 1. **Facial Expression Recognition 2013 Dataset (FER-2013)**: Base de dados com aproximadamente 30.000 imagens de faces humanas, subdivididas em 7 expressões faciais (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
 2. **TMDB 5000 Movie Dataset**: Base de dados com informações de filmes de diversos gêneros (Drama, Comedy, Thriller, Action, Romance, Adventure, Crime, Science Fiction, Horror, Family, Fantasy, Mystery, Animation, History, Music, War, Documentary, Western, Foreign, TV Movie).

Duas possíveis abordagens: Deep Learning (DL) ou Supervised Learning (SL).



Caso a abordagem por DL seja escolhida, serão utilizadas as ferramentas Pytorch e/ou TensorFlow. Caso contrário, classificadores de padrão como SVM, RF, OPF, etc.

Espera-se que o programa seja capaz de identificar a emoção em uma imagem ou foto com alta acurácia (~90%) para que o filme sugerido faça sentido.


## Cronograma
> Proposta de cronograma. Procure estimar quantas semanas serão gastas para cada etapa do projeto.

## Referências Bibliográficas
 https://docs.python.org/3/
 https://pytorch.org/tutorials/
 https://www.tensorflow.org/tutorials?hl=pt-br
 https://builtin.com/data-science/supervised-learning-python
 https://paperswithcode.com/dataset/fer2013
 https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
