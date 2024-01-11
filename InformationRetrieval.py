#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests #to make HTTP requests
from bs4 import BeautifulSoup #for web scraping purposes
import pandas as pd #for data manipulation and analysis
import time #for handling time-related operations
import datetime #for working with dates and times
import string #for string-related operations
import nltk #Natural Language Toolkit (nltk) for natural language processing tasks

import csv
import requests

# Importing urljoin from urllib.parse for constructing absolute URLs
from urllib.parse import urljoin


# In[2]:


nltk.download('averaged_perceptron_tagger') #part-of-speech tagging
nltk.download('wordnet') #WordNet, a lexical database of the English language
nltk.download('stopwords') #for common stopwords (words that are often removed in text processing)
nltk.download('punkt') #for the Punkt tokenizer model, used for tokenizing text into sentences or words


# In[3]:


from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer #for lemmatization (reducing words to their base or root form)


# In[4]:


def get_robots_crawl_delay(url):
    """
    Retrieves the crawl delay specified in the robots.txt file of a given URL.

    Parameters:
    - url (str): The URL for which the robots.txt file should be checked.

    Returns:
    - float or None: The crawl delay in seconds if specified in the robots.txt file, or None if not found or an error occurs.
    """

    # Constructing the URL for the robots.txt file by joining the base URL and '/robots.txt'
    robot_url = urljoin(url, "/robots.txt")

    # Sending an HTTP GET request to fetch the robots.txt file
    response = requests.get(robot_url)

    # Checking if the request was successful (status code 200)
    if response.status_code == 200:
        # Extracting the content of the robots.txt file
        robots_content = response.text

        # Parsing each line in the robots.txt file
        for line in robots_content.split('\n'):
            # Checking if the line starts with "Crawl-delay:"
            if line.startswith("Crawl-delay:"):
                # Extracting the crawl delay value and converting it to a float
                delay = float(line.split(":")[1].strip())

                # Returning the crawl delay in seconds
                return delay
    
    # Returning None if the crawl delay is not specified or an error occurs
    return None


# In[5]:


def scrape_publication_details(url, crawl_delay):
    """
    Scrapes publication details from a given URL, including title, authors, publication year, publication URL, and author links.

    Parameters:
    - url (str): The URL of the page containing publication details.
    - crawl_delay (float): Optional crawl delay in seconds to comply with web scraping etiquette. If None, no delay is applied.

    Returns:
    - list: A list of tuples, each containing publication details (title, authors, publication year, publication URL, author links).
    """

    # Setting a user-agent header to mimic a web browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    # Sending an HTTP GET request to the specified URL with headers
    page = requests.get(url, headers=headers)

    # Parsing the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(page.content, 'html.parser')

    # Initializing an empty list to store publication details
    publications = []

    # Looping through each publication element in the HTML
    for publication in soup.select('li.list-result-item'):
        # Extracting the title of the publication
        title_element = publication.select_one('h3.title > a')
        title = title_element.get_text(strip=True) if title_element else "Title Not Found"
        publication_url = title_element['href'] if title_element else "Publication URL Not Found"

        # Extracting the authors of the publication
        author_elements = publication.select('a.link.person')
        authors = [author.get_text(strip=True) for author in author_elements]

        # Extracting the links to author profiles
        author_links = [author['href'] for author in author_elements]

        # Extracting the publication year
        publication_year_element = publication.select_one('span.date')
        publication_year = publication_year_element.get_text(strip=True) if publication_year_element else "Publication Year Not Found"

        # Appending a tuple with publication details to the list
        publications.append((title, authors, publication_year, publication_url, author_links))

    # Applying a crawl delay if specified
    if crawl_delay:
        time.sleep(crawl_delay)

    # Returning the list of publication details
    return publications


# In[6]:


def save_to_csv(publications, csv_file):
    """
    Saves a list of publication details to a CSV file.

    Parameters:
    - publications (list): A list of tuples containing publication details.
    - csv_file (str): The path to the CSV file where the data will be saved.

    Returns:
    - None
    """

    # Opening the CSV file in write mode with specified encoding
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        # Creating a CSV writer object
        writer = csv.writer(file)

        # Writing the header row with column names
        writer.writerow(['Title', 'Authors', 'Publication_Year', 'Publication_URL', 'Author_URLs'])

        # Iterating through each publication and writing its details to the CSV file
        for title, authors, publication_year, publication_url, author_links in publications:
            # Writing a row with publication details
            writer.writerow([title, ", ".join(authors), publication_year, publication_url, ", ".join(author_links)])


# In[7]:


def main(csv_file):
    """
    Main function to crawl and scrape publication details from a specified URL, analyze the results, and save to a CSV file.

    Parameters:
    - csv_file (str): The path to the CSV file where the data will be saved.

    Returns:
    - int: The total number of publications crawled and scraped.
    """

    # URL of the page containing the publications
    url = "https://pureportal.coventry.ac.uk/en/organisations/centre-global-learning/publications/"

    # Get crawl delay from robots.txt file
    crawl_delay = get_robots_crawl_delay(url)
    if crawl_delay:
        print("Crawl delay:", crawl_delay, "seconds")

    # List to store all publications
    all_publications = []

    # Pagination variables
    current_page = 1
    count = 0

    # Dictionary to store staff and their publications
    staff_publications = {}

    # Crawl pages
    while True:
        # Scrape publication details for the current page
        result_temp = scrape_publication_details(url + f"?page={current_page-1}", crawl_delay)

        # Break if no more publications are found
        if not result_temp:
            break

        # Check for a placeholder indicating the end of pages
        if result_temp[0][0] == "Title Not Found" and not result_temp[0][1]:
            print('End Of Pages!!!\n')
            break
        else:
            # Extend the list of all publications with the current page's results
            all_publications.extend(result_temp)
            print(f"Scraped {len(result_temp)} publications from {url}?page={current_page}")
            count += len(result_temp)
            current_page += 1

            # Update staff_publications dictionary with the new results
            for _, authors, _, _, _ in result_temp:
                for author in authors:
                    if author not in staff_publications:
                        staff_publications[author] = []
                    staff_publications[author].append(_)

    # Output summary information
    print("Total number of publications:", count)
    num_staff = len(staff_publications)
    max_publications = max(len(publications) for publications in staff_publications.values())
    print("Number of staff whose publications are crawled (approximately):", num_staff)
    print("Maximum number of distinct publications per staff:", max_publications)

    # Save all publications to CSV file
    save_to_csv(all_publications, csv_file)
    print("Results saved to", csv_file, "\n")

    # Return the total number of publications
    return len(all_publications)


# In[8]:


if __name__ == "__main__":
    # Name of the CSV file to save the scraped data
    csv_file = 'coventry_university_publications.csv'

    # Calling the main function to crawl, scrape, and save publication details
    num_records = main(csv_file)

    # Reading the CSV file into a pandas DataFrame and renaming the 'Unnamed: 0' column to 'SN'
    coventry_db = pd.read_csv(csv_file).rename(columns={'Unnamed: 0': 'SN'})

    # Printing the total number of records scraped
    print(f'{coventry_db.shape[0]} records were scraped')


# In[9]:


# Add a new column 'SN' with values starting from 0 and incrementing by one for each new row
coventry_db['SN'] = range(len(coventry_db))

# Updating DataFrame back to the same file ('coventry_university_publications.csv') without writing the index
coventry_db.to_csv(csv_file, index=False)

# Display the updated DataFrame
coventry_db.head()

# Resetting index
scraped_coventry_db = coventry_db.reset_index(drop=True)
scraped_coventry_db.head()

dup_title = coventry_db["Title"]
# Finding duplicate rows
coventry_db[dup_title.isin(dup_title[dup_title.duplicated()])]

# Extract the row at index 1 and create a copy
indexone_row = coventry_db.loc[1, :].copy()

# Display the single_row DataFrame
indexone_row


# In[10]:


def preprocess_text(text):
   """
   Preprocesses the input text by converting it to lowercase, removing punctuation marks,
   and lemmatizing the words after removing stop words.

   Parameters:
       text (str): The input text to be preprocessed.

   Returns:
       str: Preprocessed text after removing stop words and lemmatizing the words.
   """
   # Convert the text to lowercase
   text = text.lower()

   # Remove punctuation marks
   text = text.translate(str.maketrans('', '', string.punctuation))

   # Lemmatize the words and remove stop words
   text = lemmatize_text(text)

   return text


# In[11]:


def get_wordnet_pos(word):
   """
   Maps the POS tag from the NLTK POS tags to WordNet POS tags.

   Parameters:
       word (str): The word for which POS tag needs to be mapped.

   Returns:
       str: WordNet POS tag.
   """
   tag = pos_tag([word])[0][1][0].upper()
   hash_tag = {"V": wordnet.VERB, "R": wordnet.ADV, "N": wordnet.NOUN, "J": wordnet.ADJ}
   return hash_tag.get(tag, wordnet.NOUN)


# In[12]:


def lemmatize_text(text):
   """
   Lemmatizes the words in the input text using WordNetLemmatizer after removing stop words.

   Parameters:
       text (str): The input text containing words to be lemmatized.

   Returns:
       str: Lemmatized text after removing stop words.
   """
   lemmatizer = WordNetLemmatizer()
   stop_words = set(stopwords.words("english"))
   tokens = nltk.word_tokenize(text)

   # Lemmatize each word and remove stop words
   lemmatized_text = ""
   for token in tokens:
       if token not in stop_words:
           lemmatized_text += lemmatizer.lemmatize(token, get_wordnet_pos(token)) + " "

   return lemmatized_text


# In[13]:


# Sample title
indexone_row['Title']

# Demonstration of lowercase and punctuation removal
preprocess_text(indexone_row['Title'])

# Demonstration of lematization
lemmatize_text(preprocess_text(indexone_row['Title']))
processed_db = scraped_coventry_db.copy()


# In[14]:


def preprocess_df(dataframe):
   """
   Preprocesses the DataFrame by applying text preprocessing to the 'Title' column,
   converting 'Authors' to lowercase, and dropping unnecessary columns.

   Parameters:
       dataframe (DataFrame): The input DataFrame to be preprocessed.

   Returns:
       DataFrame: Preprocessed DataFrame after applying the necessary transformations.
   """
   dataframe['Title'] = dataframe['Title'].apply(preprocess_text)
   dataframe['Authors'] = dataframe['Authors'].str.lower()
   dataframe = dataframe.drop(columns=['Authors', 'Publication_Year'], axis=1)
   return dataframe


# In[15]:


# Preprocess the DataFrame
preprocess_df(processed_db)
processed_db.head()

# Copy the first row of the preprocessed DataFrame for analysis
single_row = processed_db.loc[0, :].copy()

# Display the content of the single row to examine its details
print(single_row, "\n")

# Initialize an empty dictionary to keep track of the index for each word in the 'Title' column
indexing_trial = {}

# Split the 'Title' into individual words
words = single_row.Title.split()

# Obtain the 'SN' (index) of the single row
SN = single_row.SN

# Extract the first word from the 'Title' column
word = words[0]

# Create a dictionary 'word_index_dict' with the word as the key and a list containing the 'SN' as the value
word_index_dict = {word: [SN]}

# Print sample index
print('Sample index: ', word_index_dict)


# In[16]:


def index_dataframe(row_data, word_index):
   """
   Indexes the words in the 'Title' column of a DataFrame by associating them with the corresponding row indices.

   Parameters:
       row_data (pandas.Series): A single row of the DataFrame containing the 'Title' and 'SN' columns.
       word_index (dict): The word index dictionary that stores the mapping of words to their corresponding row indices.

   Returns:
       dict: The updated word index dictionary after indexing the words in the 'Title'.
   """
   # Split the 'Title' into individual words
   words = row_data.Title.split()


   # Get the 'SN' (index) of the row
   SN = int(row_data.SN)

   # Iterate through each word in the 'Title'
   for word in words:
       # If the word is already in the word index, update the index entry
       if word in word_index.keys():
           # Check if the 'SN' is not already present in the list for this word, then add it
           if SN not in word_index[word]:
               word_index[word].append(SN)
       # If the word is not in the word index, create a new entry with the 'SN'
       else:
           word_index[word] = [SN]
   return word_index


# In[17]:


# Initialize an empty dictionary to store the word index
word_index = {}

# Apply the indexing function to the 'single' row using the 'word_index' dictionary
word_index = index_dataframe(row_data=single_row, word_index={})

# Display the updated word index dictionary
print(word_index)


# In[18]:


def full_index_process(df, index):
   """
   Applies the indexing process to all rows of the DataFrame.

   Parameters:
       df (pandas.DataFrame): The DataFrame containing the 'Title' and 'SN' columns.
       index (dict): The word index dictionary to be updated.

   Returns:
       dict: The updated word index dictionary after indexing all rows.
   """
   # Iterate through each row of the DataFrame
   for x in range(len(df)):
       # Get the current row data
       row_data = df.loc[x, :]

       # Update the word index using the current row data
       index = index_dataframe(row_data=row_data, word_index=index)


   return index


# In[19]:


def construct_index_dataframe(df, index):
   """
   Constructs the word index for the given DataFrame.

   Parameters:
       df (pandas.DataFrame): The DataFrame containing the 'Title' and 'SN' columns.
       index (dict): The word index dictionary to be constructed or updated.

   Returns:
       dict: The word index dictionary containing the mapping of words to their corresponding row indices in the DataFrame.
   """
   # Preprocess the DataFrame to remove unnecessary columns and apply text preprocessing to the 'Title' column
   processed_df = preprocess_df(df)

   # Apply full indexing to the preprocessed DataFrame
   index = full_index_process(df=processed_df, index=index)

   return index


# In[20]:


# Construct the word index for the 'processed_db' DataFrame
indexed = full_index_process(df=processed_db, index={})

# Construct the word index for the 'scraped_db' DataFrame
indexes = construct_index_dataframe(df=scraped_coventry_db, index={})


# In[ ]:


import json

def save_index_to_json(index, file_path):
   """
   Save the word index to a JSON file.

   Parameters:
       index (dict): The word index.
       file_path (str): The file path to save the JSON data.
   """
   with open(file_path, 'w') as new_file:
       json.dump(index, new_file, sort_keys=True, indent=4)

def load_index_from_json(file_path):
   """
   Load the word index from a JSON file.

   Parameters:
       file_path (str): The file path from which to load the JSON data.

   Returns:
       dict: The loaded word index.
   """
   with open(file_path, 'r') as file:
       data = json.load(file)
   return data


def update_index(df, file_path):
   """
   Update the word index with new data from a DataFrame and save it to a JSON file.

   Parameters:
       df (pd.DataFrame): The DataFrame containing new data.
       file_path (str): The file path of the existing JSON file to update or create.
   """
   if len(df) > 0:
       prior_index = load_index_from_json(file_path)
       new_index = full_index_process(df=df, index=prior_index)
       save_index_to_json(new_index, file_path)

# Save the word index to 'indexes.json'
save_index_to_json(indexes, 'indexes.json')


# Load the word index from 'indexes.json'
loaded_indexes = load_index_from_json('indexes.json')

# Update the word index with new data from 'processed_db' and save it to 'indexes.json'
update_index(df=processed_db, file_path='indexes.json')
print(len(loaded_indexes))
loaded_indexes

def preprocess_query(query):
   """
   Preprocesses the user's search query by converting it to lowercase, removing punctuation marks,
   and lemmatizing the words after removing stop words.

   Parameters:
       query (str): The user's search query to be preprocessed.

   Returns:
       str: Preprocessed search query after removing stop words and lemmatizing the words.
   """

   # Convert the query to lowercase
   query = query.lower()

   # Remove punctuation marks
   query = query.translate(str.maketrans('', '', string.punctuation))

   # Lemmatize the words and remove stop words
   query = lemmatize_text(query)

   return query


def demonstrate_query_processing():
   """
   Demonstrates the query processing by taking user input for search terms,
   preprocessing the input, and displaying the processed search query.
   """
   user_input = input('Enter Search Terms: ')
   processed_query = preprocess_query(user_input)
   print(f'Processed Search Query: {processed_query}')
   return processed_query

demonstrate_query_processing()


def preprocess_and_split_query(terms):
   """
   Preprocesses the input query by converting it to lowercase, removing punctuation marks,
   lemmatizing the words after removing stop words, and then splitting the query into individual words.

   Parameters:
       terms (str): The input query to be preprocessed and split.

   Returns:
       list: A list containing individual words from the preprocessed query.
   """
   preprocessed_query = preprocess_query(terms)
   individual_words = preprocessed_query.split()
   return individual_words


def demonstrate_query_processing():
   """
   Demonstrates the query processing by taking user input for search terms,
   preprocessing the input, and displaying the processed search query.
   """
   user_input = input('Enter Search Terms: ')
   processed_query = preprocess_query(user_input)
   print(f'Processed Search Query: {processed_query}')
   return processed_query

# Get the preprocessed and split query
dqp = demonstrate_query_processing()
split_query_result = preprocess_and_split_query(dqp)

print(f'Split Query: {split_query_result}')


def get_union(lists):
   """
   Computes the union of multiple lists and returns the result as a sorted list.

   Parameters:
       lists (list): A list of lists to find the union.

   Returns:
       list: The union of all lists as a sorted list.
   """
   union = list(set.union(*map(set, lists)))
   union.sort()
   return union


def get_intersection(lists):
   """
   Computes the intersection of multiple lists and returns the result as a sorted list.
   Parameters:
       lists (list): A list of lists to find the intersection.

   Returns:
       list: The intersection of all lists as a sorted list.
   """
   intersect = list(set.intersection(*map(set, lists)))
   intersect.sort()
   return intersect




def vertical_search_engine(data_frame, query, word_index):
   """
   Perform a vertical search on the given DataFrame using the provided word index for the query.

   Parameters:
       data_frame (pd.DataFrame): The DataFrame to search.
       query (str): The search query to be processed.
       word_index (dict): The word index containing mapping of words to SNs.

   Returns:
       pd.DataFrame or str: The search result DataFrame if matches are found, or 'No result found' if no matches.
   """
   query_terms = preprocess_and_split_query(query)  # Split the query into individual terms
   retrieved_sns = []

   # Retrieve SNs for each term in the query from the word index
   for term in query_terms:
       if term in word_index:
           retrieved_sns.append(word_index[term])

   # Perform Ranked Retrieval if matches are found
   if len(retrieved_sns) > 0:
       high_rank_result = get_intersection(retrieved_sns)  # High-rank result is the intersection of retrieved SNs
       low_rank_result = get_union(retrieved_sns)  # Low-rank result is the union of retrieved SNs
       uncommon_sns = [x for x in low_rank_result if x not in high_rank_result]
       high_rank_result.extend(uncommon_sns)
       result_sns = high_rank_result

       # Extract the final output DataFrame containing the search result
       final_output = data_frame[data_frame.SN.isin(result_sns)].reset_index(drop=True)

       # Merge the result DataFrame with the SNs to maintain the order of Intersection ----> Union
       dummy = pd.Series(result_sns, name='SN').to_frame()
       result_df = pd.merge(dummy, final_output, on='SN', how='left')
   else:
       result_df = 'No result found'

   return result_df

def test_search_engine(data_frame, word_index):
   """
   Perform a test search on the given DataFrame using the provided word index.

   Parameters:
       data_frame (pd.DataFrame): The DataFrame to search.
       word_index (dict): The word index containing mapping of words to SNs.

   Returns:
       pd.DataFrame or str: The search result DataFrame if matches are found, or 'No result found' if no matches.
   """
   query = input("Enter your search query: ")
   result = vertical_search_engine(data_frame, query, word_index)
   return result

def final_engine(results):
   """
   Display the final search results in a formatted manner.

   Parameters:
       results (pd.DataFrame or str): The search results DataFrame or 'No result found' message.

   Returns:
       None
   """
   if isinstance(results, pd.DataFrame):
       for i in range(len(results)):
           printout = results.loc[i, :]
           print(f"Title: {printout['Title']}")
           print(f"Author: {printout['Authors']}")
           print(f"Published: {printout['Publication_Year']}")
           print(f"Link: {printout['Publication_URL']}")
           print(f"Author Link: {printout['Author_URLs']}")
           print('')
   else:
       print(results)


def test_search_engine(df, index):
   query = input("Enter your search query: ")
   return vertical_search_engine(df, query, index)


# Test the search engine and display the results
results = test_search_engine(scraped_coventry_db, indexed)
final_engine(results)

# Set the initial value for days and interval
days = 0
interval = 7

# Run the loop indefinitely
while True:
   scrape_publication_details("https://pureportal.coventry.ac.uk/en/organisations/centre-global-learning/publications/", interval)
   print(f"Crawled at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
   print(f'Next crawl scheduled after {interval} days')
   time.sleep(interval * 24 * 60 * 60)  # Convert days to seconds

