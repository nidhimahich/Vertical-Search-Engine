This code is a web crawler that extracts publication information from the website "https://pureportal.coventry.ac.uk/en/organisations/centre-global-learning/publications/." 

### 1. **Getting Ready: Imports**
   - **What it does:** We brought in special tools to help our system work well. It's like getting the right equipment before starting a job.
   - **Examples:**
     - `requests`: It helps us get information from web pages.
     - `BeautifulSoup`: This tool helps us understand and extract useful things from web pages.
     - `pandas`: It's like a magic wand for handling data - making it easy to analyze and organize.
     - `nltk` (Natural Language Toolkit): It's our language understanding helper.

### 2. **Web Explorer (Crawler)**
   - **What it does:** We built a program that acts like a smart explorer on the internet. It goes to a university webpage, looks for publications, and saves the important details.
   - **Functions:**
     - `get_robots_crawl_delay(url)`: Checks if the webpage has any rules for how often our explorer should visit.
     - `scrape_publication_details()`: The explorer does the actual work of finding and saving publication information.
     - `save_to_csv()`: It organizes the gathered information into a nice file.

### 3. **Sorting and Cataloging**
   - **What it does:** After exploring, we make sure all the information is neatly arranged. Imagine putting books in a library so you can find them easily.
   - **Actions:**
     - Adding an 'SN' column: It's like giving each publication a special number.
     - Updating Dataframe: Keeping everything in order, like arranging books on a shelf.

### 4. **Cleaning Up Text**
   - **What it does:** We created tools to clean up the words in titles and other details. It's like fixing typos and making everything look nice.
   - **Functions:**
     - `preprocess_text(text)`: This tool changes the words to lowercase and removes unnecessary punctuation.
     - `get_wordnet_pos(word)`: It helps us understand the role of a word, like whether it's a verb or a noun.
     - `lemmatize_text(text)`: This tool simplifies words so we can understand them better.

### 5. **Making Sense of Words**
   - **What it does:** We taught our system to understand what words mean and how they fit together. It's like explaining grammar to a computer.
   - **Tasks:**
     - Part-of-Speech Tagging: Understanding if a word is a noun, verb, etc.
     - Lemmatization: Simplifying words to their basic form.

### 6. **Putting it All Together (Search Engine)**
   - **What it does:** We built a special search engine using all these tools. It helps us find information quickly and efficiently.
   - **Functions:**
     - `vertical_search_engine(data_frame, query, word_index)`: The search engine uses a query (like a question) and a tool (word index) to find the right information.
     - `test_search_engine(data_frame, word_index)`: It's like trying out the search engine to make sure it works.
     - `final_engine(results)`: This function makes the final results look nice and easy to understand.

### 7. **Scheduled Web Explorer (Crawler)**
   - **What it does:** This is like having our explorer check the webpage regularly, like a friendly reminder. It ensures we always have the most recent information.
   - **Actions:**
     - Infinite Loop: Our explorer keeps checking without stopping.
     - Crawling Interval: It waits for a set amount of time between checks.

That's the gist of our system. It's like having a smart helper that explores the web, organizes information, understands language, and quickly finds what you're looking for!
