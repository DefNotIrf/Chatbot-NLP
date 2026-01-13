import csv
import json

CSV_PATH = "data/siddiq_menu.csv"
JSON_PATH = "data/siddiq_menu.json"

menu_data = []

with open(CSV_PATH, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        menu_data.append(row)

with open(JSON_PATH, "w", encoding="utf-8") as jsonfile:
    json.dump(menu_data, jsonfile, indent=4)

print("CSV successfully converted to JSON")
