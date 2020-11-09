# Pandas Introduction
import pandas as pd

# Dataframe basics
f500 = pd.read_csv('f500.csv',index_col=0)
f500.index.name = None
f500_type = type(f500)
f500_shape = f500.shape

f500_head = f500.head(6)
f500_tail = f500.tail(8)

# Prints information about dataframe (shape, dtypes)
f500.info()

# Data types of dataframe columns
f500_dtypes = f500.dtypes

# select column from data frame - series
industries = f500.loc[:, "industry"]
# or shortcut
industries = f500["industry"]
industries_type = type(industries)

countries = f500["country"]
# Several columns
revenues_years = f500[["revenues", "years_on_global_500_list"]]
# Column Slicing
ceo_to_sector = f500.loc[:, "ceo":"sector"]

# Row series
toyota = f500.loc["Toyota Motor"]
# Several rows
drink_companies = f500.loc[["Anheuser-Busch InBev", "Coca-Cola", "Heineken Holding"]]
# Row and col slicing
middle_companies = f500.loc["Tata Motors":"Nationwide", "rank":"country"]
# Shortcut
middle_companies_all = f500["Tata Motors":"Nationwide"]

# Counts of values in series
countries = f500["country"]
countries_counts = countries.value_counts()

# Slicing series (.loc can be omitted for shortcut)
india = countries_counts.loc["India"]
north_america = countries_counts.loc[["USA", "Canada", "Mexico"]]

big_movers = f500.loc[["Aviva", "HP", "JD.com", "BHP Billiton"], ["rank", "previous_rank"]]
bottom_companies = f500.loc["National Grid": "AutoNation", ["rank", "sector", "country"]]

# Data manipulation
f500_head = f500.head()
f500_head.info()

# Vectorized operations
rank_change = f500["previous_rank"] - f500["rank"]
rank_change_max = rank_change.max()
rank_change_min = rank_change.min()

# Statistics
rank = f500["rank"]
rank_desc = rank.describe()

prev_rank = f500["previous_rank"]
prev_rank_desc = prev_rank.describe()

# Select count of value == 0.
zero_previous_rank = f500["previous_rank"].value_counts().loc[0]

# Statistics on dataframes (along index (0) or columns (1) axis). This example uses only numeric columns
# axis == index by default, can be omitted
max_f500 = f500.max(axis="index", numeric_only = True)

# Assign values
f500.loc["Dow Chemical", "ceo"] = "Jim Fitterling"

# Boolean indexing
motor_bool = f500["industry"] == "Motor Vehicles and Parts"
motor_countries = f500.loc[motor_bool, "country"]

# Replace zero value with NaN in the column
prev_rank_before = f500["previous_rank"].value_counts(dropna=False).head()
f500.loc[f500["previous_rank"] == 0, "previous_rank"] = np.nan
prev_rank_after = f500["previous_rank"].value_counts(dropna=False).head()

# Adding new column to dataframe
f500["rank_change"] = f500["previous_rank"] - f500["rank"]
rank_change_desc = f500["rank_change"].describe()

# Selecting top values for countries
industry_usa = f500.loc[f500["country"] == "USA", "industry"].value_counts().head(2)
sector_china = f500.loc[f500["country"] == "China", "sector"].value_counts().head(3)