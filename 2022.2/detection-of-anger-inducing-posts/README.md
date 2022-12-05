# `Detecção de posts indutores de raiva`
# `Detection of anger-inducing posts`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA930 - Computação Afetiva*, 
oferecida no segundo semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

|Nome  | RA | Especialização|
|--|--|--|
| Douglas Esteves  | 189697  | Eng. de Computação  | 
| Pedro Fracarolli | 191535  | Analista de Sistemas|


## Descrição Resumida do Projeto

Veículos de mídia e redes sociais têm interesse em aumentar o engajamento de seus usuários em conteúdos publicados, afim de aumentar a probabilidade de que os mesmos gerem interações com anúncios e, portanto, mais lucro. Uma das formas de se obter esse aumento de engajamento é através de conteúdo que gere raiva ou desconforto nos usuários, de forma que os mesmos se sintam compelidos a agir através de comentários, desta forma promovendo algo que seja de interesse de anunciantes.

O objetivo deste trabalho é tentar detectar posts que possam conter gatilhos emocionais negativos, de forma a alertar os leitores sobre a possível real intenção da notícia, com foco em métodos de processamento de linguagem natural, num primeiro momento. 

> [IA930- Detecção de posts indutores de raiva](https://youtu.be/FMxbsorAHPg)


## Metodologia Proposta

- Tentaremos treinar um modelo (ou utilizar algum pré-treinado - provavelmente [BERT](https://arxiv.org/abs/1810.04805) ou algo similar) no [dataset GoEmotions](https://ai.googleblog.com/2021/10/goemotions-dataset-for-fine-grained.html).

- Usaremos o dataset [Facebook News](https://github.com/jbencina/facebook-news) para tentarmos detectar posts com reações majoritariamente de raiva. 

- É esperado que consigamos detectar com certa precisão posts que contenham gatilhos emocionais negativos que levem os usuários a reagir em posts com raiva.

- De forma alternativa, tentaremos gerar comentários para posts e detectar as emoções para tentar ter uma assertividade maior, mas isso ainda está para ser definido.

- Idealmente iremos desenvolver uma extensão para o navegador Mozilla Firefox que utilize o modelo para fazer as classificações enquanto o usuário navega pelas redes sociais.

- Inicialmente iremos usar Python, PyTorch e modelos de linguagem disponíveis na HuggingFace. Vamos avaliar implementações em TensorFlow também, principalmente porque pretendemos usar o modelo em uma extensão de browser.

## Cronograma

|Semana|17/10|24/10|31/10|07/11|14/11|21/11|28/11|05/12|
|--|--|--|--|--|--|--|--|--|
|Revisão bibliográfica|X|X|
|Desenvolvimento/treinamento do modelo|||X|X|X
|Avaliação|||||X
|Ajustes no modelo para classificação de comentários|||||X|X|X|
|Desenvolvimento da extensão para Firefox|||||||X|X|
|Relatório final|||||||X|X|


## Referências Bibliográficas
- [GoEmotions: A Dataset of Fine-Grained Emotions](https://arxiv.org/abs/2005.00547)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
