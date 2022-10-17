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

Modelos generativos como GAN, VAE, baseados em fluxo mostraram grande sucesso na geração de amostras de alta qualidade, mas possuem algumas limitações próprias. 
Os modelos GAN e VAE são conhecidos por treinamento potencialmente instável e menor diversidade na geração devido à sua natureza de treinamento contraditório.
Por outro lado, os modelos baseados em fluxo precisam usar arquiteturas especializadas para construir a transformada reversível.
Os modelos de difusão que serão utilizados neste projeto propõem superar as limitações dos modelos generativos mencionados. Eles definem uma cadeia de Markov de etapas de difusão, nas quais é lentamente adicionado ruído aleatoriario as imagens de entrada. Depois, o modelo aprende a reverter o processo de difusão para construir novas imagens a partir do ruído. Neste projeto, vamos explorar os modelos de difusão com o objetivo de gerar imagens faciais que expressam uma emoção, já que esse enfoque não tem sido implementado. Para isso, serão utilizadas bases de dados que contêm imagens faciais que foram etiquetadas com uma emoção específica. Em conclusão, o nosso projeto estará focado na geração condicional de rostos humanos emocionais usando modelos de difusão. 

Vídeo da proposta do projeto: 

## Metodologia Proposta

A metodologia do projeto será explicada nas seguintes seções.

### Bases de dados

Visando gerar rostos de pessoas que expressam alguma emoção, precisamos de bases de dados etiquetadas usando modelos de emoções. Como primeira perspectiva do projeto, temos duas opções de bases de dados: FER2013 Dataset [1] e Extended Cohn-Kanade Dataset (CK+) [2]. A continuação apresentamos uma breve descrição de cada base de dados.

#### FER2013 Dataset 

Esta base de dados tem sido amplamente utilizada para reconhecimento de emoções faciais, e está disponível publicamente em Kaggle. FER2013 contem 35887 imagens normalizadas a 48x48 pixeis em escala de cinza, que expressam 7 emoções: raiva, nojo, medo, felicidade, tristeza, surpresa, neutro. 

Enlace do repositório: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

<p align="center">
	<img src="https://github.com/kgrosero/ia930/blob/main/2022.2/emo-diffusion/figures/fer2013.png" align="middle" width="700">
	  <figcaption>
  	Figura 1: Amostras da base de dados FER2013.
  	</figcaption>
</p>

#### Extended Cohn-Kanade Dataset (CK+)

Esta base de dados contem 593 sequências de images de 123 pessoas na faixa etaria entre 18 e 50 anos. Cada sequência de imagens tem entre 10 e 60 quadros da pessoa em transição da emoção neutra para a emoção alvo. As 7 emoções consideras são: raiva, nojo, medo, felicidade, tristeza, surpresa, e desprezo. Cada quadro tem 640x480 pixeis em escala de cinza ou com cor. 

Enlace do repositório: https://www.kaggle.com/datasets/shawon10/ckplus

<p align="center">
	<img src="https://github.com/kgrosero/ia930/blob/main/2022.2/emo-diffusion/figures/ck%2B.png" align="middle" width="700">
	  <figcaption>
  	Figura 2: Amostras da base de dados CK+.
  	</figcaption>
</p>

### Modelos de difusão

Os modelos generativos são uma classe de métodos de aprendizado de máquina que aprendem uma representação dos dados em que são treinados e modelam os próprios dados. Eles geralmente são baseados em redes neurais profundas.  Os modelos generativos permitem sintetizar novos dados que são diferentes dos dados reais, mas ainda parecem tão realistas. É assim que se treinarmos um modelo generativo em imagens de carros, a modelo seria capaz de gerar imagens de novos carros com aparências diferentes.

Recentemente, os modelos de difusão surgiram como uma poderosa classe de métodos de aprendizagem generativa. Esses modelos, também conhecidos como modelos de difusão de denoising ou modelos generativos baseados em pontuação, demonstram uma qualidade de amostra surpreendentemente alta, muitas vezes superando as redes adversárias generativas. Os modelos de difusão já foram aplicados a uma variedade de tarefas de geração, como imagens e fala.

Os modelos de difusão consistem em dois processos: difusão direta e reversa parametrizada. Como mostrado na Figura 3, o processo de difusão direta mapeia dados para ruído perturbando gradualmente os dados de entrada. Isso é formalmente alcançado por um processo estocástico simples que começa a partir de uma amostra de dados e gera iterativamente amostras mais ruidosas usando um kernel de difusão gaussiana simples. Ou seja, em cada etapa desse processo, o ruído gaussiano é adicionado aos dados de forma incremental. O segundo processo é um processo reverso parametrizado que desfaz a difusão direta e realiza a remoção de ruído iterativa. Este processo representa a síntese de dados e é treinado para gerar dados convertendo ruído aleatório em dados realistas [3]. Tanto o processo direto quanto o reverso geralmente usam milhares de etapas para injeção gradual de ruído e durante a geração para eliminação de ruído [4].

<p align="center">
	<img src="https://github.com/kgrosero/ia930/blob/main/2022.2/emo-diffusion/figures/diff2.png" align="middle" width="700">
	  <figcaption>
  	Figura 3: Metodologia do algoritmo de difusão.
  	</figcaption>
</p>

Neste projeto, o modelo de geração utilizado será o modelo de difusão, com o objetivo de gerar condicionalmente rostos de pessoas que evocam uma determinada emoção.
Além disso, este projeto aborda a perspectiva de geração condicionada, que neste caso sera baseada nos rótulos da classe que representam um grupo de emoções. 

### Ferramentas 

Para este projeto será utilizado Google Colab Pro. O código base dos modelos de difusão foi desenvolvido em PyTorch e está disponível neste repositório https://github.com/NVlabs/denoising-diffusion-gan.

### Resultados esperados 

reinar modelos generativos em imagens com informações de condicionamento, como o conjunto de dados ImageNet, é comum gerar amostras condicionadas em rótulos de classe ou em um texto descritivo.

<p align="center">
	<img src="https://github.com/kgrosero/ia930/blob/main/2022.2/emo-diffusion/figures/results.png" align="middle" width="700">
	  <figcaption>
  	Figura 4: Exemplo dos resultados esperados.
  	</figcaption>
</p>


### Proposta de avaliação

Segundo [4], os modelos de difusão podem ser avaliados usando a função log de verossimilhança, em que o procedimento de treinamento visa melhorar o amostrador dinâmico de Langevin usando inferência variacional. Portanto, a função log de verossimilhança será a métrica objetiva a ser utilizada no projeto. 

Também propomos usar uma avaliação subjetiva da qual os colegas da disciplina poderiam ser parte, com o objetivo de saber se a imagem gerada seria considerada por pessoas, como pertencendo à categoria alvo.

## Cronograma

A continuação detalhamos as atividades a serem desenvolvidas nas próximas semanas até o dia da apresentação final do projeto.

<b> 24/10 </b>: Processamento de base de dados 1

<b> 31/10 </b>: Processamento de base de dados 2

<b> 7/11 </b>: Estudo de modelos de difusão

<b> 14/11, 21/11 </b>: Adaptação do código dos modelos de difusão com as bases de dados propostas

<b> 28/11 </b>: Avaliação dos resultados e escrita do relatório do projeto.

<b> 5/12 </b>: Apresentação do projeto


## Referências Bibliográficas

[1] I. J. Goodfellow, D. Erhan, P. L. Carrier, A. Courville, M. Mirza, B.
Hamner, W. Cukierski, Y. Tang, D. Thaler, D.-H. Lee et al.,
“Challenges in representation learning: A report on three machine
learning contests,” in International Conference on neural Information Processing. Springer, 2013, pp. 117-124.

[2] P. Lucey, J. F. Cohn, T. Kanade, J. Saragih, Z. Ambadar and I. Matthews, "The Extended Cohn-Kanade Dataset (CK+): A complete dataset for action unit and emotion-specified expression," 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition - Workshops, 2010, pp. 94-101, doi: 10.1109/CVPRW.2010.5543262.

[3] Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.

[4] Jonathan Ho et al. “Denoising diffusion probabilistic models.” arxiv Preprint arxiv:2006.11239 (2020).

[4] Alex Nichol & Prafulla Dhariwal. “Improved denoising diffusion probabilistic models” arxiv Preprint arxiv:2102.09672 (2021).

[6] Aditya Ramesh et al. “Hierarchical Text-Conditional Image Generation with CLIP Latents." arxiv Preprint arxiv:2204.06125 (2022).
