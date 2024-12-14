import kagglehub
import os

# Download the dataset
path = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")

# Move the files Books_rating.csv and book_data.csv to the ./data/
os.rename(os.path.join(path, "Books_rating.csv"), "./data/Books_rating2.csv")
os.rename(os.path.join(path, "books_data.csv"), "./data/books_data2.csv")
