# Book Recommender System

## I. Project description

The  goal of this project is to build a recommendation engine that aims to help users find books which might be interesting for them based on their summaries. We'll do this by applying Latent Dirichlet Allocation - LDA algorithm.

> **Latent Dirichlet Allocation** is a type of unobserved learning algorithm in which topics are inferred from a dictionary of text corpora whose structures are not known (are latent). The practical use of such an algorithm is to solve the cold-start problem, whereby analytics can be done on texts to derive similarities in the dictionary's corpses, and further be expanded to others outside the original scope - all without user input, which is often an early stumbling block for businesses that do not meet early usage quotas to derive needed analysis.

The data for this project (all books on Wikipedia) has been collected from Wikipedia dumps in compressed form from May 1, 2020. Using the technology outlined by Will Koehrsen of Medium's Towards Data Science, we use the process of the XML processing program to separate the individual pages, the article title, the content of the article and the length of each article.

Next is an overview of the work related to this topic, and then the theoretical explanation behind LDA to lay the foundation for our implementation. Finally, we will study the complete code and introduce the specific results of the algorithm, and discuss some of its shortcomings.

## II. Topic Modeling

We want to build a recommender system in order to recommend books. For example, if I read a Sherlock Holmes book, the system would recommend me equivalent books like the Orient Express or the Murder at the Vicarage. To do this, we would like to extract features (topics) to obtain something like:

- **The Adventures of Sherlock Holmes** = 60% Detective + 30% Adventure + 10% Horror
- **Murder on the Orient Express** = 62% Detective + 33% Adventure + 5% Horror

And we compute the distance between extracted features. We can use either Euclidian distance (the lower the better) or Cosine similarity (the higher the better)

- **Euclidian distance:** $||a - b||_2 = \sqrt{\sum_i{(a_i - b_i)^2}} \approx 0.004$
- **Cosine similarity:** $\frac{a^Tb}{||a||||b||} = cos(a, b) \approx 0.997$

Then, we can define a document (book) as a *distribution over topics*. We can also define a topic as a *distribution over words*.

Eg.
- **Sport:** 20% Football + 10% Hockey + 5% Goal + 1% Score + ...
- **Economy:** 24% Money + 9% Dollar + 7% Euro + 3% Bank + ...
- **Politics:** 10% President + 4% USA + 3% Union + 1% Law + ...

So, the goal of topic modeling is to *construct topics*, and *assign them to texts*.

## III. Latent Dirichlet Allocation

## IV. Implementation

## References
* Will Koehrsen - Wikipedia Data Science: Working with the World’s Largest Encyclopedia. [link](https://towardsdatascience.com/wikipedia-data-science-working-with-the-worlds-largest-encyclopedia-c08efbac5f5c)
* Tyler Doll - LDA Topic Modeling: An Explanation. [link](https://towardsdatascience.com/lda-topic-modeling-an-explanation-e184c90aadcd)
