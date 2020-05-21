# Book Recommender System

## I. Project description

The  goal of this project is to build a recommendation engine that aims to help users find books which might be interesting for them based on their summaries. We'll do this by applying Latent Dirichlet Allocation - LDA algorithm.

> **Latent Dirichlet Allocation** is a type of unobserved learning algorithm in which topics are inferred from a dictionary of text corpora whose structures are not known (are latent). The practical use of such an algorithm is to solve the cold-start problem, whereby analytics can be done on texts to derive similarities in the dictionary's corpses, and further be expanded to others outside the original scope - all without user input, which is often an early stumbling block for businesses that do not meet early usage quotas to derive needed analysis.

The data for this project (all books on Wikipedia) has been collected from Wikipedia dumps in compressed form from May 1st, 2020. Using the technology outlined by Will Koehrsen of Medium's Towards Data Science, we use the process of the XML processing program to separate the individual pages, the article title, the content of the article and the length of each article.

Next is an overview of the work related to this topic, and then the theoretical explanation behind LDA to lay the foundation for our implementation. Finally, we will study the complete code and introduce the specific results of the algorithm, and discuss some of its shortcomings.

## II. Topic Modeling

We want to build a recommender system in order to recommend books. For example, if I read a Sherlock Holmes book, the system would recommend me equivalent books like the Orient Express or the Murder at the Vicarage. To do this, we would like to extract features (topics) to obtain something like:

- **The Adventures of Sherlock Holmes** = 60% Detective + 30% Adventure + 10% Horror
- **Murder on the Orient Express** = 62% Detective + 33% Adventure + 5% Horror

And we compute the distance between extracted features. We can use either Euclidian distance (the lower the better) or Cosine similarity (the higher the better)

- **Euclidian distance:** <img src="/tex/ad9868ef7c30d901e5bec5b2fd6aaf24.svg?invert_in_darkmode&sanitize=true" align=middle width=251.69588399999998pt height=28.602918299999985pt/>
- **Cosine similarity:** <img src="/tex/b425dc497ecc8e039a1a27b074cc46f8.svg?invert_in_darkmode&sanitize=true" align=middle width=186.01759109999998pt height=34.099002299999995pt/>

Then, we can define a document (book) as a *distribution over topics*. We can also define a topic as a *distribution over words*.

Eg.
- **Sport:** 20% Football + 10% Hockey + 5% Goal + 1% Score + ...
- **Economy:** 24% Money + 9% Dollar + 7% Euro + 3% Bank + ...
- **Politics:** 10% President + 4% USA + 3% Union + 1% Law + ...

So, the goal of topic modeling is to *construct topics*, and *assign them to texts*.

## III. Latent Dirichlet Allocation

The general idea of LDA is that **each document is generated from a mixture of topics and each of those topics is a mixture of words.**. LDA is formed of:
- **Latent**: Topic structures in a document are latent meaning they are hidden structures in the text.
- **Dirichlet**: The Dirichlet distribution determines the mixture proportions of the topics in the documents and the words in each topic.
- **Allocation**: Allocation of words to a given topic.

LDA models the probability distribution below:

<p align="center"><img src="/tex/ea985ffdaefb8c055c4c482af44746d4.svg?invert_in_darkmode&sanitize=true" align=middle width=359.94013275pt height=48.4659351pt/></p>

with:

<p align="center"><img src="/tex/6548fd73a53f2f905b86b73e22b7b6bc.svg?invert_in_darkmode&sanitize=true" align=middle width=113.88606735pt height=16.438356pt/></p>

<p align="center"><img src="/tex/2c0331d3c221430274e86b90bf9021ea.svg?invert_in_darkmode&sanitize=true" align=middle width=132.0935682pt height=19.0044921pt/></p>

<p align="center"><img src="/tex/fcdf631b3d324741d6e17b61f9191216.svg?invert_in_darkmode&sanitize=true" align=middle width=171.8442264pt height=18.18259905pt/></p>

and constraints:

<p align="center"><img src="/tex/117d95ef1b16d8ac1d425a61056d76bb.svg?invert_in_darkmode&sanitize=true" align=middle width=161.75383455pt height=36.16460595pt/></p>

Where:
- <img src="/tex/78ec2b7008296ce0561cf83393cb746d.svg?invert_in_darkmode&sanitize=true" align=middle width=14.06623184999999pt height=22.465723500000017pt/>: documents
- <img src="/tex/5e5baf9603d1d171daa4a81f176f6a6b.svg?invert_in_darkmode&sanitize=true" align=middle width=20.050844549999987pt height=22.465723500000017pt/>: words in document <img src="/tex/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55596444999999pt height=22.831056599999986pt/>
- <img src="/tex/d665a4d2867c2875db92529de77c4b3e.svg?invert_in_darkmode&sanitize=true" align=middle width=111.8963472pt height=24.65753399999998pt/>: topic for each words
- <img src="/tex/3d078f9fed536c6801aa65e4926dd8d8.svg?invert_in_darkmode&sanitize=true" align=middle width=117.55272045pt height=24.65753399999998pt/>: words
- <img src="/tex/134ed36f05e9862e9910f5b0dc9620f7.svg?invert_in_darkmode&sanitize=true" align=middle width=41.004093899999994pt height=24.65753399999998pt/>: generate topic proba.
- <img src="/tex/7ea8749aaad38e93ea0ecddfc7364de6.svg?invert_in_darkmode&sanitize=true" align=middle width=72.51815009999999pt height=24.65753399999998pt/>: select topic
- <img src="/tex/eb7103ff61aae57fe104302a9de8d2dd.svg?invert_in_darkmode&sanitize=true" align=middle width=88.4524113pt height=24.65753399999998pt/>: select word from topic

**Known:** <img src="/tex/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode&sanitize=true" align=middle width=17.80826024999999pt height=22.465723500000017pt/> - data

**Unknown:** <img src="/tex/5e16cba094787c1a10e568c61c63a5fe.svg?invert_in_darkmode&sanitize=true" align=middle width=11.87217899999999pt height=22.465723500000017pt/> (params, dist. over words for each topic), <img src="/tex/5b51bd2e6f329245d425b8002d7cf942.svg?invert_in_darkmode&sanitize=true" align=middle width=12.397274999999992pt height=22.465723500000017pt/> (latent var., topic for each word), <img src="/tex/b35e24d8a08c0ab01195f2ad2a78fab7.svg?invert_in_darkmode&sanitize=true" align=middle width=12.785434199999989pt height=22.465723500000017pt/> (latent var., dist. over topics for ech doc.)

## IV. Implementation

## References
* Will Koehrsen - Wikipedia Data Science: Working with the World’s Largest Encyclopedia. [link](https://towardsdatascience.com/wikipedia-data-science-working-with-the-worlds-largest-encyclopedia-c08efbac5f5c)
* Tyler Doll - LDA Topic Modeling: An Explanation. [link](https://towardsdatascience.com/lda-topic-modeling-an-explanation-e184c90aadcd)
