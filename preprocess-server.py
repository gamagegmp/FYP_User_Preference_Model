import pandas as pd
import ast

file_path = "Dataset.csv"

# Load the dataset
df = pd.read_csv(file_path)
print("Dataset loaded successfully!")

# --------------------------------------------------- Data Preprocessing ---------------------------------------------------
# Create a copy of the dataframe
df_filled = df.copy()

# --------------------------------------------------- Remove Unnecessary Columns ---------------------------------------------------
remove_columns = [
    'Rating_0', 'Rating_1', 'Rating_2', 'Rating_3', 'Rating_4', 'Rating_5', 
    'Rating_6', 'Rating_7', 'Rating_8', 'Rating_9', 'Rec_0', 'Rec_1', 'Rec_2',
    'Rec_3', 'Rec_4', 'Rec_5', 'Rec_6', 'Rec_7', 'Rec_8', 'Rec_9', 'no_swipes',
    'maybe_swipes', 'yes_swipes', 'Model', 'Retrieval', 'DynaMatch', 'form_h', 
    'form_i', 'form_j', 'form_k','form_r'
]

df_filled = df_filled.drop(columns=remove_columns)

# --------------------------------------------------- Fill Missing Categorical Columns with Mode ---------------------------------------------------
categorical_columns = ['form_a', 'form_b', 'form_c', 'form_f', 'form_g', 'form_rr']
df_filled[categorical_columns] = df_filled[categorical_columns].apply(lambda x: x.fillna(x.mode()[0]))

# --------------------------------------------------- Convert Multiple Selections into Binary Columns ---------------------------------------------------
def convert_multiple_selection_column(df, column_name, values, new_column_prefix):
    """Convert list-like string columns into multiple binary columns"""
    df[column_name] = df[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    for value in values:
        new_column_name = f"{new_column_prefix}_{value}"
        df[new_column_name] = df[column_name].apply(lambda x: 1 if isinstance(x, list) and str(value) in x else 0)

    df = df.drop(columns=[column_name])
    return df

# Apply one-hot encoding
df_filled = convert_multiple_selection_column(df_filled, 'form_a', [0, 1, 2, 3], 'age_range')
df_filled = convert_multiple_selection_column(df_filled, 'form_b', [0, 1, 2, 3], 'budget_range')
df_filled = convert_multiple_selection_column(df_filled, 'form_c', [0, 1, 2, 3], 'season')
df_filled = convert_multiple_selection_column(df_filled, 'form_f', [0, 1, 2, 3, 4, 5, 6, 7], 'experience')
df_filled = convert_multiple_selection_column(df_filled, 'form_g', [0, 1, 2, 3, 4, 5, 6, 7], 'scenery')
df_filled = convert_multiple_selection_column(df_filled, 'form_rr', ['e','n','c','a','s','m','f','o'], 'preferred_region')


# Rename columns
df_filled.rename(columns={
    "id": "user_id",
    "age_range_0": "age_0_19",
    "age_range_1": "age_20_39",
    "age_range_2": "age_40_59",
    "age_range_3": "age_60_plus",
    "budget_range_0": "budget_0_49",
    "budget_range_1": "budget_50_99",
    "budget_range_2": "budget_100_249",
    "budget_range_3": "budget_300_plus",
    "season_0": "season_winter",
    "season_1": "season_spring",
    "season_2": "season_summer",
    "season_3": "season_fall",
    "experience_0": "experience_beach",
    "experience_1": "experience_adventure",
    "experience_2": "experience_nature",
    "experience_3": "experience_culture",
    "experience_4": "experience_nightlife",
    "experience_5": "experience_history",
    "experience_6": "experience_shopping",
    "experience_7": "experience_cuisine",
    "scenery_0": "scenery_urban",
    "scenery_1": "scenery_rural",
    "scenery_2": "scenery_sea",
    "scenery_3": "scenery_mountain",
    "scenery_4": "scenery_lake",
    "scenery_5": "scenery_desert",
    "scenery_6": "scenery_plains",
    "scenery_7": "scenery_jungle",
    "preferred_region_e": "preferred_region_europe",
    "preferred_region_n": "preferred_region_n_america",
    "preferred_region_c": "preferred_region_caribbean",
    "preferred_region_a": "preferred_region_asia",
    "preferred_region_s": "preferred_region_s_america",
    "preferred_region_m": "preferred_region_mid_east",
    "preferred_region_f": "preferred_region_africa",
    "preferred_region_o": "preferred_region_oceania",
}, inplace=True)

# Change the user Id format to integers
df_filled['user_id'] = range(1, len(df_filled) + 1)

# --------------------------------------------------- Final Check ---------------------------------------------------
print(df_filled.head())

df_filled.to_csv("Preprocessed_Dataset.csv", index=False)
print("Preprocessed dataset saved successfully!")
