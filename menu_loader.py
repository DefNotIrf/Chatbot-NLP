import pandas as pd

def load_menu_data(csv_path="data/siddiq_menu.csv"):
    return pd.read_csv(csv_path)

def retrieve_menu(df, stall_name):
    return df[df["stall_name"].str.lower().str.contains(stall_name.lower())]

def menu_to_text(df):
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"Meal: {row['meal_name']}, "
            f"Tags: {row['tags']}, "
            f"Price: {row['price']} ringgit, "
            f"Available at: {row['available_time']}."
        )
    return "\n".join(lines)
