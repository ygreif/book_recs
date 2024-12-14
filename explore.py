import matplotlib.pyplot as plt
from data import data

df = data.df('./data/Books_rating.csv')

# Histogram for review scores
df['review/score'].hist(bins=5, edgecolor='black')
plt.title('Distribution of Review Scores')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

df.groupby('Id')['review/score'].mean().hist(bins=5, edgecolor='black')
plt.title('Distribution of Average Review Scores')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()
