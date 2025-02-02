# Objective:

# To analyze the dataset and derive actionable insights regarding the sources of leads (e.g., Instagram, Facebook, peer-to-peer, etc.) and the nature of their interest (e.g., interested farmer, want to buy plants, etc.).

# Step 1: Data Understanding and Preparation

# Load the Data:

    # Use Python's pandas library to load the .xlsx file.

    # Example: import pandas as pd; df = pd.read_excel('data.xlsx')

# Inspect the Data:

    # Check the structure of the dataset using df.head(), df.info(), and df.describe().

    # Identify missing values, duplicates, and inconsistencies in columns like Reference/From, Comments, etc.


# Step 2: Exploratory Data Analysis (EDA)

# Analyze Lead Sources (Reference/From):

    # Use value_counts() to identify the most common sources of leads.

    # Visualize the distribution using a bar chart or pie chart (e.g., using matplotlib or seaborn).

    # Example: df['Reference/From'].value_counts().plot(kind='bar')

# Analyze Comments/Interest Levels:

    # Categorize comments into broader groups (e.g., "Interested in Buying Plants," "Seeking Information," "Undecided").

    # Use groupby to analyze the relationship between lead sources and interest levels.

    # Example: df.groupby(['Reference/From', 'Comments']).size().unstack().plot(kind='bar', stacked=True)

# Temporal Analysis (if applicable):

    # Analyze trends over time using the date column.

    # Example: df['date'].value_counts().sort_index().plot(kind='line')

# Step 3: Insights and Recommendations

# Key Insights:

    # Identify the most effective lead sources (e.g., Instagram, Facebook, stalls).

    # Determine the most common types of interest (e.g., buying plants, seeking information).

    # Highlight any trends or patterns over time.

# Recommendations:

    # Allocate more resources to high-performing lead sources (e.g., increase social media ad spend).

    # Tailor communication strategies based on interest levels (e.g., provide detailed information to undecided leads).

    # Address gaps in underperforming lead sources (e.g., improve stall engagement).


import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel('data.xlsx')

# Analyze lead sources
lead_sources = df['Reference/From'].value_counts()
lead_sources.plot(kind='bar', title='Lead Sources Distribution')
plt.show()

# Analyze comments
df['Comments'] = df['Comments'].str.lower()  # Standardize comments
comments_summary = df['Comments'].value_counts()
comments_summary.plot(kind='pie', autopct='%1.1f%%', title='Interest Levels')
plt.show()

# Group by source and comments
grouped_data = df.groupby(['Reference/From', 'Comments']).size().unstack()
grouped_data.plot(kind='bar', stacked=True, title='Lead Sources vs Interest Levels')
plt.show()