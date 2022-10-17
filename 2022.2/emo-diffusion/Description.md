# `<Explorando modelos de difusão para gerar imagens de rostros emocionais>`
# `<Exploring diffusion models to generate images of emotional faces>`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA930 - Computação Afetiva*, 
oferecida no segundo semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> Incluir nome RA e foco de especialização de cada membro do grupo. Os grupos devem ter no máximo três integrantes.
> |Nome  | RA | Especialização|
> |--|--|--|
> | Karen Rosero  | 264373  | Eng. em Eletrônica e Telecomunicações|
> | Renan Yamaguti  | 262731  | Eng. em Computação|

## Descrição Resumida do Projeto
> Descrição do tema do projeto, incluindo contexto gerador, motivação.
> Descrição do objetivo principal do projeto.
> 
Modelos generativos como GAN, VAE, baseados em fluxo mostraram grande sucesso na geração de amostras de alta qualidade, mas possuem algumas limitações próprias. 
Os modelos GAN e VAE são conhecidos por treinamento potencialmente instável e menor diversidade na geração devido à sua natureza de treinamento contraditório.
Por outro lado, os modelos baseados em fluxo precisam usar arquiteturas especializadas para construir a transformada reversível.
Os modelos de difusão que serão utilizados neste projeto propõem superar as limitações dos modelos generativos mencionados. Eles definem uma cadeia de Markov de etapas de difusão, nas quais é lentamente adicionado 
ruído aleatoriario as imagens de entrada. Depois, o modelo aprende a reverter o processo de difusão para construir novas imagens a partir do ruído. 
Neste projeto, vamos explorar os modelos de difusão com o objetivo de gerar imagens faciais que expressam uma emoção. Para isso, serão utilizadas bases de dados que contêm 
imagens faciais que foram etiquetadas com uma emoção específica. O nosso projeto estará focado na geração condicional de rostos humanos emocionais. 

> Incluir nessa seção link para vídeo de apresentação da proposta do projeto (máximo 10 minutos).

## Metodologia Proposta
> Para a primeira entrega, a metodologia proposta deve procurar responder:
> * Qual(is) base(s) de dado(s) o projeto pretende utilizar, justificando a(s) escolha(s) realizadas.
> * Quais abordagens o grupo já enxerga como interessantes de serem estudadas.
> * Artigos de referência já identificados e que serão estudados ou usados como parte do planejamento do projeto
> * Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).
> * Resultados esperados
> * Proposta de avaliação

## Cronograma
> Proposta de cronograma. Procure estimar quantas semanas serão gastas para cada etapa do projeto.

## Referências Bibliográficas

[1] Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.
[2] 