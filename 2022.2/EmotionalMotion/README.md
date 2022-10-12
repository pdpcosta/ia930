# Detectando emoções pelo celular para sugestão de músicas.

# Emotion recognition by smartphone for music suggestion.

  

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA930 - Computação Afetiva*, 

oferecida no segundo semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

|Nome  | RA | Especialização|
|--|--|--|
| Alexsandro Ferreira de Barros Júnior | 233768  | Eng. Eletricista|
| Bruno Guedes da Silva  | 203657  | Eng. Eletricista|
| Davi Pereira Da Silva | 233429  | Eng. Eletricista|

## Descrição Resumida do Projeto

Dado o contexto pós pandêmico, constatou-se o aumento da procura por atendimento de saúde mental (3,4) em função do sofrimento advindo dos traumas nesse período caótico. Sendo assim, busca-se através desse projeto obter uma forma de auxiliar, não clinicamente, no tratamento desses traumas valendo-se da tecnologias cotidianas.  

Assim, baseado nas experiências sobre influência da música à resposta emocional do ser humano (1,2), propõe-se a detecção de estado emocional do usuário através de sensores de smartphone ou smartwatch para regulação emocional a partir da sugestão de músicas . 

Como celular está no cotidiano de muitas pessoas, uma aplicação que apoie a regulação emocional dos usuários através deste dispositivo teria a capacidade de sugerir momentos de relaxamento em qualquer circunstância do dia, contribuindo para a saúde mental e física dos mesmos.

O objetivo principal do projeto consiste na implementação de um sistema de dois módulos principais: 

1. Detector de estado emocional a partir de dados de acelerômetro e giroscópio (eventualmente outros dados sensoriais poderão ser acrescentados);  
2.  Módulo de sugestão de música para regulação de estados emocionais alterados.

Link Vídeo: 

## Metodologia Proposta

O projeto será desenvolvido nas seguintes etapas:

1.  Levantamento de dados  
    1. Enumeração e escolha de datasets de dados inerciais (acelerômetro e giroscópio) com classificação do estado emocional durante captura. Dois datasets iniciais foram considerados:
        - SMARTED (5)
        - K-EmoPhone (6)
2.  Desenvolvimento do módulo de detecção de estados emocionais   
    1.  Inicialmente será explorado técnicas de aprendizado de máquina com baixo custo computacional, facilitando a implementação em celulares
3.  Desenvolvimento do módulo de sugestão de músicas
    1.  Levantamento de músicas classificadas pelo Geneva Emotion Music Scales (1) para obter banco de dados de músicas a serem sugeridas.
    2.  Estudo de métodos para relacionar o impacto das músicas na regulação emocional. 
    3.  Implementação de sistema de regras para sugestão de músicas
4.  Integração da aplicação
5.  Validação e Avaliação do sistema
    1.  Detector de estado emocional será validado com métricas de avaliação de classificadores (acurácia, precisão, recall, …) a partir dos datasets utilizados
    2.  A aplicação será avaliada qualitativamente e quantitativamente (se possível) através de testes de uso e questionários realizados com os demais colegas de turma

## Cronograma

| Etapa | Semana 1 | Semana 2 | Semana 3 | Semana 4 | Semana 5 | Semana 6 | Semana 7 | Semana 8 |
|--|--|--|--|--|--|--|--|--|
|Proposta do Projeto| X | | | | | | | |
|Levantamento de dados | X | X | | | | | | |
| Módulo de Detecção Emocional | | X | X | X | | | | |
| Módulo de Sugestão | | | X | X | X | | | |
| Integração da Aplicação | | | |  | X | X | X | |
| Validação e Avaliação | | | | | | | X | X |
|Relatório Final| | | | | | | | X |

## Referências Bibliográficas

(1) Zentner, Marcel, Didier Grandjean, and Klaus R. Scherer. **"**Emotions evoked by the sound of music: characterization, classification, and measurement.** Emotion 8.4 (2008): 494.  Disponível em: https://psycnet.apa.org/record/2008-09984-007

(2) Simões, Ana Rita Chichorro. **As Emoções ao compasso da música: um olhar sobre a influência da música na resposta emocional.** Diss. 2012. Disponível em: https://repositorio.ul.pt/handle/10451/8076

(3) Conceição, Ana e Felipe Frisch. **Pandemia aumenta procura por atendimento de saúde mental**. 2021. Disponível em: https://valor.globo.com/brasil/noticia/2021/04/19/pandemia-aumenta-procura-por-atendimento-de-saude-mental.ghtml  
(4) Menon, Isabelle. "Pandemia levou a aumento na busca por terapia e lotou agendas". 2022.  Disponível em: https://www1.folha.uol.com.br/equilibrioesaude/2022/04/pandemia-levou-a-aumento-na-busca-por-terapia-e-lotou-agendas.shtml

(5) PEPA, L. et al. **SMARTED: SMARTwatch Emotion Dataset**. Zenodo, , 25 jul. 2022. Disponível em: https://zenodo.org/record/6900984 

(6) KANG, S. et al. **K-EmoPhone, A Mobile and Wearable Dataset with In-Situ Emotion, Stress, and Attention Labels**. Zenodo, , 3 ago. 2022. Disponível em: https://zenodo.org/record/6900984#.Y0cPTnbMJEZ
