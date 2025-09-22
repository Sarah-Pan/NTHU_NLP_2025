# %% [markdown]
# ## Part I: Data Pre-processing

# %%
import pandas as pd

# %%
# Download the Google Analogy dataset
!wget http://download.tensorflow.org/data/questions-words.txt
# I download by the link instead

# %%
# Preprocess the dataset
file_name = "questions-words"
with open(f"{file_name}.txt", "r") as f:
    data = f.read().splitlines()

# %%
# check data from the first 10 entries
for entry in data[:10]:
    print(entry)

# %%
# TODO1: Write your code here for processing data to pd.DataFrame
# Please note that the first five mentions of ": " indicate `semantic`,
# and the remaining nine belong to the `syntatic` category.

questions = []
categories = []
sub_categories = []

category_count = 0
current_subcat = None
current_category = None

for line in data:
    if not line:
        continue
    if line.startswith(":"):
        current_subcat = line
        category_count += 1
        if category_count <= 5:
            current_category = "Semantic"
        else:
            current_category = "Syntactic"
    else:
        questions.append(line)
        categories.append(current_category)
        sub_categories.append(current_subcat)


# df = pd.DataFrame(records, columns=["Question", "Category", "SubCategory"])
# df.head()


# %%
# Create the dataframe
df = pd.DataFrame(
    {
        "Question": questions,
        "Category": categories,
        "SubCategory": sub_categories,
    }
)

# %%
df.head()

# %%
df.to_csv(f"{file_name}.csv", index=False)

# %% [markdown]
# ## Part II: Use pre-trained word embeddings
# - After finish Part I, you can run Part II code blocks only.

# %%
import pandas as pd
import numpy as np
import gensim.downloader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# %%
data = pd.read_csv("questions-words.csv")

# %%
MODEL_NAME = "glove-wiki-gigaword-100"
# You can try other models.
# https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models

# Load the pre-trained model (using GloVe vectors here)
model = gensim.downloader.load(MODEL_NAME)
print("The Gensim model loaded successfully!")

# %%
# Do predictions and preserve the gold answers (word_D)
preds = []
golds = []

for analogy in tqdm(data["Question"]):
      # TODO2: Write your code here to use pre-trained word embeddings for getting predictions of the analogy task.
      # You should also preserve the gold answers during iterations for evaluations later.
      """ Hints
      # Unpack the analogy (e.g., "man", "woman", "king", "queen")
      # Perform vector arithmetic: word_b + word_c - word_a should be close to word_d
      # Source: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L776
      # Mikolov et al., 2013: big - biggest and small - smallest
      # Mikolov et al., 2013: X = vector(”biggest”) − vector(”big”) + vector(”small”).
      """
      a, b, c, d = analogy.split()
      golds.append(d.lower())

      if any(w.lower() not in model.wv.key_to_index for w in (a, b, c)):
          preds.append(None)
          continue

      try:
          # b + c - a
          pred = model.wv.most_similar(
              positive=[b.lower(), c.lower()],
              negative=[a.lower()],
              topn=1
          )[0][0]
      except KeyError:
          pred = None
      preds.append(pred)


# %%
# Perform evaluations. You do not need to modify this block!!

def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(gold == pred)

golds_np, preds_np = np.array(golds), np.array(preds)
data = pd.read_csv("questions-words.csv")

# Evaluation: categories
for category in data["Category"].unique():
    mask = data["Category"] == category
    golds_cat, preds_cat = golds_np[mask], preds_np[mask]
    acc_cat = calculate_accuracy(golds_cat, preds_cat)
    print(f"Category: {category}, Accuracy: {acc_cat * 100}%")

# Evaluation: sub-categories
for sub_category in data["SubCategory"].unique():
    mask = data["SubCategory"] == sub_category
    golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
    acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
    print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")

# %%
# Collect words from Google Analogy dataset
SUB_CATEGORY = ": family"

# TODO3: Plot t-SNE for the words in the SUB_CATEGORY `: family`
family_df = df[df["SubCategory"] == SUB_CATEGORY]

words = set()
for analogy in family_df["Question"]:
    a, b, c, d = analogy.split()
    words.update([a, b, c, d])
words = list(words)

vectors = []
valid_words = []
for w in words:
    try:
        vectors.append(model[w])
        valid_words.append(w)
    except KeyError:
        continue

vectors = np.array(vectors)
tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=5)
embeddings = tsne.fit_transform(vectors)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1])

for i, word in enumerate(valid_words):
    plt.annotate(word, (embeddings[i, 0], embeddings[i, 1]))

plt.title("Word Relationships from Google Analogy Task")
plt.show()
plt.savefig("word_relationships.png", bbox_inches="tight")

# %% [markdown]
# ### Part III: Train your own word embeddings

# %% [markdown]
# ### Get the latest English Wikipedia articles and do sampling.
# - Usually, we start from Wikipedia dump (https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2). However, the downloading step will take very long. Also, the cleaning step for the Wikipedia corpus ([`gensim.corpora.wikicorpus.WikiCorpus`](https://radimrehurek.com/gensim/corpora/wikicorpus.html#gensim.corpora.wikicorpus.WikiCorpus)) will take much time. Therefore, we provide cleaned files for you.

# %%
# Download the split Wikipedia files
# Each file contain 562365 lines (articles).
!gdown --id 1jiu9E1NalT2Y8EIuWNa1xf2Tw1f1XuGd -O wiki_texts_part_0.txt.gz
!gdown --id 1ABblLRd9HXdXvaNv8H9fFq984bhnowoG -O wiki_texts_part_1.txt.gz
!gdown --id 1z2VFNhpPvCejTP5zyejzKj5YjI_Bn42M -O wiki_texts_part_2.txt.gz
!gdown --id 1VKjded9BxADRhIoCzXy_W8uzVOTWIf0g -O wiki_texts_part_3.txt.gz
!gdown --id 16mBeG26m9LzHXdPe8UrijUIc6sHxhknz -O wiki_texts_part_4.txt.gz

# %%
# Download the split Wikipedia files
# Each file contain 562365 lines (articles), except the last file.
!gdown --id 17JFvxOH-kc-VmvGkhG7p3iSZSpsWdgJI -O wiki_texts_part_5.txt.gz
!gdown --id 19IvB2vOJRGlrYulnTXlZECR8zT5v550P -O wiki_texts_part_6.txt.gz
!gdown --id 1sjwO8A2SDOKruv6-8NEq7pEIuQ50ygVV -O wiki_texts_part_7.txt.gz
!gdown --id 1s7xKWJmyk98Jbq6Fi1scrHy7fr_ellUX -O wiki_texts_part_8.txt.gz
!gdown --id 17eQXcrvY1cfpKelLbP2BhQKrljnFNykr -O wiki_texts_part_9.txt.gz
!gdown --id 1J5TAN6bNBiSgTIYiPwzmABvGhAF58h62 -O wiki_texts_part_10.txt.gz

# %%
# Extract the downloaded wiki_texts_parts files.
!gunzip -k wiki_texts_part_*.gz

# %%
# Combine the extracted wiki_texts_parts files.
!cat wiki_texts_part_*.txt > wiki_texts_combined.txt

# %%
# Check the first ten lines of the combined file
!head -n 10 wiki_texts_combined.txt

# %% [markdown]
# Please note that we used the default parameters of [`gensim.corpora.wikicorpus.WikiCorpus`](https://radimrehurek.com/gensim/corpora/wikicorpus.html#gensim.corpora.wikicorpus.WikiCorpus) for cleaning the Wiki raw file. Thus, words with one character were discarded.

# %%
# Now you need to do sampling because the corpus is too big.
# You can further perform analysis with a greater sampling ratio.

import random

wiki_txt_path = "wiki_texts_combined.txt"
output_path = "wiki_texts_sampled.txt"
# wiki_texts_combined.txt is a text file separated by linebreaks (\n).
# Each row in wiki_texts_combined.txt indicates a Wikipedia article.

sample_ratio = 0.2



with open(wiki_txt_path, "r", encoding="utf-8") as f:
    with open(output_path, "w", encoding="utf-8") as output_file:
        # TODO4: Sample `20%` Wikipedia articles
        # Write your code here
        total, kept = 0, 0
        for line in f:
            total += 1
            if random.random() < sample_ratio:
                output_file.write(line)
                kept += 1


# %%
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from tqdm import tqdm

# 1. Load the sampled articles
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

sampled_articles_path = "wiki_texts_sampled.txt"

# 2. pre-processing before training
class MyCorpus:
    def __init__(self, file_path):
        self.file_path = file_path
        # count lines for tqdm
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.num_lines = sum(1 for _ in f)

    def __iter__(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=self.num_lines, desc="Pre-processing"):
                # 1. tokenize + lower-case + exclude punctuation
                tokens = simple_preprocess(line)

                # 2. stopwords, non-English words exclusion
                tokens = [w for w in tokens if w.isalpha() and w not in stop_words]

                # 3. lemmatization (rocks -> rock, dogs -> dog)
                tokens = [lemmatizer.lemmatize(w) for w in tokens]

                if tokens:
                    yield tokens

corpus = MyCorpus(sampled_articles_path)

# 3. Train Word2Vec model
model = Word2Vec(
    sentences=corpus,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,
    epochs=5
)

# 4. Save the trained model
model.save("wiki_word2vec.model")

# %%
data = pd.read_csv("questions-words.csv")

# %%
# Do predictions and preserve the gold answers (word_D)
preds = []
golds = []

for analogy in tqdm(data["Question"]):
      # TODO2: Write your code here to use pre-trained word embeddings for getting predictions of the analogy task.
      # You should also preserve the gold answers during iterations for evaluations later.
      """ Hints
      # Unpack the analogy (e.g., "man", "woman", "king", "queen")
      # Perform vector arithmetic: word_b + word_c - word_a should be close to word_d
      # Source: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L776
      # Mikolov et al., 2013: big - biggest and small - smallest
      # Mikolov et al., 2013: X = vector(”biggest”) − vector(”big”) + vector(”small”).
      """
      a, b, c, d = analogy.split()
      golds.append(d.lower())

      if any(w.lower() not in model.wv.key_to_index for w in (a, b, c)):
          preds.append(None)
          continue

      try:
          # b + c - a
          pred = model.wv.most_similar(
              positive=[b.lower(), c.lower()],
              negative=[a.lower()],
              topn=1
          )[0][0]
      except KeyError:
          pred = None
      preds.append(pred)


# %%
# Perform evaluations. You do not need to modify this block!!

def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(gold == pred)

golds_np, preds_np = np.array(golds), np.array(preds)
data = pd.read_csv("questions-words.csv")

# Evaluation: categories
for category in data["Category"].unique():
    mask = data["Category"] == category
    golds_cat, preds_cat = golds_np[mask], preds_np[mask]
    acc_cat = calculate_accuracy(golds_cat, preds_cat)
    print(f"Category: {category}, Accuracy: {acc_cat * 100}%")

# Evaluation: sub-categories
for sub_category in data["SubCategory"].unique():
    mask = data["SubCategory"] == sub_category
    golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
    acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
    print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")

# %%
# Collect words from Google Analogy dataset
SUB_CATEGORY = ": family"

# TODO7: Plot t-SNE for the words in the SUB_CATEGORY `: family`
# 1. Filter the dataframe for the chosen sub-category
family_df = data[data["SubCategory"] == SUB_CATEGORY]

# 2. collect unique words
words = set()
for analogy in family_df["Question"]:
    a, b, c, d = analogy.split()
    words.update([a, b, c, d])
words = list(words)

# 3. get their vectors from the trained Word2Vec model
vectors = []
valid_words = []
for w in words:
    if w in model.wv:   # make sure the word is in the vocabulary
        vectors.append(model.wv[w])
        valid_words.append(w)

vectors = np.array(vectors)

# 4. t-SNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=5)
embeddings = tsne.fit_transform(vectors)

# 5. Plotting
plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1])

for i, word in enumerate(valid_words):
    plt.annotate(word, (embeddings[i, 0], embeddings[i, 1]))


plt.title("Word Relationships from Google Analogy Task")
plt.show()
plt.savefig("word_relationships.png", bbox_inches="tight")


