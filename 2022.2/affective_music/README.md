# Geração de imagens afetivas a partir de dados de música
# Affective images generator from music data

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA930 - Computação Afetiva*, 
oferecida no segundo semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

 |Nome  | RA | Especialização|
 |--|--|--|
 | Pedro Mendes Odilon  | 204708  | Eng. Eletricista|
 | Matheus Gimenez Fernandes  | 240543  | Eng. Físico|
 | Sara Sousa de Oliveira  | 205733  | Eng. Eletricista|

## Descrição Resumida do Projeto

Este projeto se dedica a geração de imagens a partir de áudio, usando como formas de transferência modelos emocionais, texto ,features do áudio etc. Dentre as motivações do problema estão; criação de capas de álbuns de música automaticamente e acessibilidade para surdos.

Objetivo: gerar imagem a partir de música

> Incluir nessa seção link para vídeo de apresentação da proposta do projeto (máximo 10 minutos).

## Metodologia Proposta
 Para a primeira entrega, a metodologia proposta deve procurar responder:
 * Qual(is) base(s) de dado(s) o projeto pretende utilizar, justificando a(s) escolha(s) realizadas.
    * https://cvml.unige.ch/databases/DEAM/
 * Quais abordagens o grupo já enxerga como interessantes de serem estudadas.
    * Pensamos em pegar rótulos de emoções, combinar com aspectos semânticos e dados dimensionais afetivos da música para gerar uma string que será a entrada para a geração de uma imagem no software Dall-e 2. 
 * Artigos de referência já identificados e que serão estudados ou usados como parte do planejamento do projeto
   * TALAMINI, Francesca et al. Musical emotions affect memory for emotional pictures. Scientific reports, v. 12, n. 1, p. 1-8, 2022.
   * ZHAO, Sicheng et al. Emotion-based end-to-end matching between image and music in valence-arousal space. In: Proceedings of the 28th ACM International Conference on Multimedia. 2020. p. 2945-2954.
   * CHEN, Chin-Han et al. Emotion-based music visualization using photos. In: International Conference on Multimedia Modeling. Springer, Berlin, Heidelberg, 2008. p. 358-368.
 * Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).
   * Python 
   * Librosa
   * Dall-e 2 (https://github.com/lucidrains/DALLE2-pytorch)
 * Resultados esperados
   * Conseguir sintetizar uma imagem a partir de features afetivas e aspectos semânticos de uma música.
 * Proposta de avaliação
   * Avaliar o quão boa ficou uma imagem gerada a partir de uma música. Perguntar para pessoas e pedir uma avaliação da qualidade dentro de uma escala de 0 a 10, para critérios de desempenho.
