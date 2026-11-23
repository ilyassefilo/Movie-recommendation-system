# Movie Recommendation System

Before we start,It's worth mentioning that this was an implementation of the same project that was published on youtube by Dataquest.
Type a movie title to find it in the MovieLens catalogue, then discover what else its fans enjoyed. The notebook connects two techniques. **TF-IDF and cosine similarity** match the words you type to movie titles. **Collaborative filtering** uses viewer ratings to rank movies that are especially popular among fans of the selected film.

For example, when the starting film is *Toy Story (1995)*, the recommendations include *Toy Story 2* and *A Bug’s Life*. The first shares words with the original title. The second makes the list because of how the same viewers rated it.

## Run the notebook

1. Download and extract the [MovieLens 25M dataset from GroupLens](https://grouplens.org/datasets/movielens/25m/). Place `movies.csv` and `ratings.csv` beside `movie recommendation system(2).ipynb`. The notebook reads both files from its working directory.
2. From that directory, install the dependencies and start JupyterLab.

   ```bash
   python -m pip install jupyterlab pandas numpy scikit-learn ipywidgets
   python -m jupyterlab
   ```

3. Open the notebook and run the cells from top to bottom. Loading the 25 million ratings can take a moment.
4. Try the first **Movie Title** box to see five title matches. In the final box, edit the prefilled title to request recommendations. Both boxes respond to a change once the input contains at least six characters. Try typing `Toy Story 1995` to include the release year.

To reproduce the worked example for the original *Toy Story* exactly, run this in a cell after the function definitions:

```python
find_similar_movies(1)[["title", "genres", "similar", "all", "score"]]
```

## What I implemented

### 1. Search the catalogue by title

The notebook loads movie IDs, titles and genres with pandas. I keep the original title for display and create a second column, `clean_title`, for searching. A regular expression removes characters outside ASCII letters, digits and spaces, so `Toy Story (1995)` becomes `Toy Story 1995`. Keeping the year helps distinguish films with the same name.

I use `TfidfVectorizer(ngram_range=(1, 2))` to turn each cleaned title into a numeric vector. I fit the vectorizer once on the catalogue. For each new query, `search(title)` applies the same cleaning and transforms it using the vocabulary and weights learned from those titles. Cosine similarity then measures the match between the query and each title. NumPy selects five candidates, and pandas retrieves their original rows.

### 2. Recommend movies from viewer ratings

The notebook first works through the calculations for movie ID `1`, then wraps them in `find_similar_movies(movie_id)` so the same process works for another film.

1. Find the unique viewers who gave the starting movie **5 stars**. For *Toy Story*, the saved example found **13,506 viewers**.
2. Collect the movie IDs those viewers rated **above 4 stars**. Count each ID with `value_counts()` and divide by the number of viewers in the group. This gives the `similar` column.
3. Keep movies that **more than 10%** of the group rated above 4. For *Toy Story*, **92 movies** passed this filter.
4. Calculate a comparison rate for each candidate. The `all` column uses everyone who rated **at least one candidate** above 4 as its denominator. For each movie, the numerator is the number of those viewers who rated that particular movie above 4.
5. Calculate `score = similar / all`, sort by score in descending order, and join the top ten rows to the movie catalogue by `movieId` to display titles and genres.

### 3. Connect the two stages in Jupyter

I used `ipywidgets.Text` to collect the title and `widgets.Output` to display a fresh table each time the input changes. The final widget takes the first row returned by `search(title)`, reads its `movieId`, and passes that ID to `find_similar_movies()`.

## What I learned

| Concept | How I understood it through this project |
| --- | --- |
| Term frequency (TF) | Each title counts as one document. TF counts how many times a term appears in that title. In *Toy Story*, both words occur once. |
| Inverse document frequency (IDF) | IDF measures how widespread a term is across different titles. A common word such as *the* carries less weight than a term that appears in only a few titles. |
| TF-IDF | Multiplying TF by IDF gives each term a weight. Scikit-learn then normalizes the title vector to unit length. This lets a distinctive shared term contribute more to a match than a common word. |
| Unigrams and bigrams | A unigram is one word and a bigram is a pair of adjacent words. The query *Toy Story* has three search features: `toy`, `story`, and `toy story`. The bigram captures the words appearing together in that order. |
| Cosine similarity | It compares the directions of two vectors. With these nonnegative weights, a score of 1 means the vectors point in the same direction and 0 means they share no weighted features. |
| Collaborative filtering | Viewers can connect two movies even when their titles share no words. Here, giving the same starting film five stars defines the group whose preferences I examine. |
| Relative popularity | A movie that many people love can have a high fan-group percentage. Dividing by its comparison rate shows how strongly it stands out for this particular group. |

One pandas detail mattered throughout: selecting `["movieId"]` **before** calling `value_counts()` makes the unit of counting explicit. Calling it on the entire ratings table counts combinations of user, movie, rating and timestamp instead.

## Reading the results

These rows come from the notebook’s saved example for *Toy Story (1995)*. The two fractions are shown as percentages here for readability.

| Movie | Fans who rated it above 4 (`similar`) | Comparison rate (`all`) | Score |
| --- | ---: | ---: | ---: |
| Toy Story (1995) | 100.00% | 12.58% | 7.95 |
| Toy Story 2 (1999) | 29.55% | 5.42% | 5.45 |
| A Bug’s Life (1998) | 12.47% | 2.53% | 4.93 |

For *Toy Story 2*, dividing 29.55% by 5.42% gives about **5.45**. In this dataset, a high rating for the sequel was about 5.45 times as common among the original film’s five-star fans as in the comparison group. The starting film itself appears because every member of its fan group gave it five stars.

## Interpreting the project’s limits

- **Title selection matters.** The search uses `np.argpartition` to select five candidates. Reversing those rows does not guarantee similarity order, yet the final widget automatically uses the first one. Queries with no recognized terms can also return arbitrary candidates. Calling the recommender with a known movie ID removes this ambiguity.
- **The score measures observed ratings.** An unrated film contributes no positive rating, and the 10% threshold favors films seen by enough fans. A score of 5.45 is a ratio of observed rates, not a predicted star rating.
- **Genres provide context.** They appear in the output so readers can interpret the recommendations. The ranking itself uses ratings, and the notebook reports example results without a held-out accuracy evaluation.
