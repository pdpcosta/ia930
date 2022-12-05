# `Detecção de posts indutores de raiva`
# `Detection of anger-inducing posts`
 
## Apresentação
O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA930 - Computação Afetiva*, oferecida no segundo semestre de 2022, na Unicamp, sob supervisão da Professora Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).
 
|Nome  | RA | Especialização|
|--|--|--|
| Douglas Esteves  | 189697  | Eng. de Computação  | 
| Pedro Fracarolli | 191535  | Analista de Sistemas|
 
## Descrição do Projeto:
 
Veículos de mídia e redes sociais têm interesse em aumentar o engajamento de seus usuários em conteúdos publicados, a fim de aumentar a probabilidade de que os mesmos geram interações com anúncios e, portanto, mais lucro. Uma das formas de se obter esse aumento de engajamento é através de conteúdo que gere raiva ou desconforto nos usuários, de forma que os mesmos se sintam compelidos a agir através de comentários, desta forma promovendo algo que seja de interesse dos anunciantes.
 
O objetivo deste trabalho é tentar detectar posts que possam conter gatilhos emocionais negativos, de forma a alertar os leitores sobre a possível real intenção da notícia, com foco em métodos de processamento de linguagem natural, num primeiro momento. 
 
[IA930- Detecção de posts indutores de raiva](https://youtu.be/FMxbsorAHPg)
 
## Abordagem Adotada
A proposta foi utilizar o treinamento de um modelo para classificação de texto em 3 níveis diferentes de conteúdo agressivo (com raiva) no dataset Facebook News.

Gerando uma avaliação da acurácia do modelo em um conjunto de testes do mesmo dataset.
Utilizamos a arquitetura: BERT sigla de Bidirectional Enconder Representations from Transformers (representações de codificador bidirecional de transformadores) (fine-tuning no FBNews).

Foi proposto um plugin que utilizasse o modelo para fazer as classificações online, esse desenvolvimento do plugin para o firefox ficou para a pŕoxima etapa. 
 
Modelo BERT: 
Baseado em transformers (encoders).
Através de duas tarefas de pré-treinamento: MLM (classificação multilabel) e NSP (classificação binária).
Após o pré-treinamento, foi realizado o fine-tuning (treinamento na tarefa alvo). 
 
Tokenizers
Traduz as palavras do vocabulário em word ids, que são passadas ao modelo para treinamento ou inferência.
Podem ser simples expressões regulares, mas em arquiteturas mais modernas usam tokenizers treinados.
No nosso trabalho, usamos um Tokenizer pré-treinado para o modelo que escolhemos.
 
 
## Resultados Finais
 
Classificação dos post de acordo com a quantidade de reações de raiva dos mesmo.
Menor que 1% em post inofensivo.
 - Entre 1% e 10% em post moderadamente agressivo.
 - Mais que 10% em post agressivo.
 
Fizemos também o fine-tuning do modelo bert-base-cased, levando em consideração palavras maiúsculas e minúsculas).
Divisão do dataset entre treino 80%, validação 20% e teste 20%.
Baixa acurácia do modelo final 36% no conjunto de teste.

Utilização de outro modelo BERT pré-treinado com fine-tuing no dataset GoEmotions para classificar os comentários dos posts do FBNews.
Contagem da quantidade de posts classificados como “anger” por comentário.

Comentários agressivos considerados como reações agressivas.

Utilização da mesma metodologia anterior. 

Resultados também ruins:  34,6% de acurácia no conjunto de testes.

## Notebooks 

- [FBNews_classification.ipynb](https://github.com/EstevesDouglas/ia930/blob/main/2022.2/notebooks/FBNews_classification.ipynb)

- [Model_Training.ipynb](https://github.com/EstevesDouglas/ia930/blob/main/2022.2/notebooks/Model_Training.ipynb)
 
## Discussão
 
Possíveis explicações
Tokenizer: usamos um pré-treinado que talvez não capture o “zeitgeinst”.

Não é possível detectar sentimentos através de manchetes somente pois as mesmas usam linguagem mais neutra (apesar de despertarem sentimentos em seres humanos).

No caso dos comentários, pode ser que os comentários usados no GoEmotions seguem uma distribuição diferente daqueles do FBNews.
 
Foram utilizadas as seguintes ferramentas durante o projeto:
- Python
- Pytorch (para DL).
- Huggingface Transformes (implementação do BERT e Tokenizer, bem como repositório de modelos pré-treinados).
- PyTorch Lightning - framework criado em cima do PyTorch para facilitar implementação de loops de treinamento, validação e teste.
- JavaScript (plugin).
 
## Trabalhos Futuros
 
Treinar um Tokenizer específico para a tarefa em questão.
 
Treinar um BERT do zero (MLM) em um dataset mais específico, para ter uma nova avaliação mais profunda e entender qual melhor estratégia que funcionam melhor para tipos de interações humanas. 
 
Trabalho futuro sobre o plugin no navegador.
Duas formas: colocar o modelo em um servidor, através de um torchserver e fazer o plugin acessá-lo, ou “embarcar” o modelo no próprio plugin.
O primeiro requer infraestrutura extra, o segundo pode ser proibitivo dependendo do tamanho do modelo (espaço, tamanho do pacote, memória, processador para inferência).
 
## SLIDES
- [Detecção de Posts Indutores de Raiva](https://docs.google.com/presentation/d/1djM6LmyLw2U8rIfPvfilG2f1DnUYxYYxdP-9m0REGek/edit#slide=id.p)
 
## Referências Bibliográficas
 
- [GoEmotions: A Dataset of Fine-Grained Emotions](https://arxiv.org/abs/2005.00547)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Long Short-term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
- [Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary)
