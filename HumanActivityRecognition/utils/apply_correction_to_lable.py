import os
import pandas as pd

"""Script per correggere le etichette nei CSV dei giocattoli
(ball, car, elephant, doll) usando un file Excel con 4 sheet.

- Legge gli intervalli da "toys_lable_correction_indices.xlsx"
- Per ogni CSV indicato, azzera 'action' e 'action_id'
  nelle righe [row_start, row_end] specificate
- I CSV vengono sovrascritti nella cartella indicata
"""

# === CONFIG ===
EXCEL_PATH = "toys_lable_correction_indices.xlsx"  
CSV_FOLDER = "."  
SHEETS = ["BALL", "CAR", "ELEPHANT", "DOLL"]  

# === FUNCTION CORRECTION ===
def apply_corrections(sheet_name, df_corrections):
    for _, row in df_corrections.iterrows():
        csv_file = os.path.join(CSV_FOLDER, row["file_name"])
        if not os.path.exists(csv_file):
            print(f"File non trovato: {csv_file}")
            continue


        df = pd.read_csv(csv_file)

        #zero based
        start = int(row["row_start"]) 
        end = int(row["row_end"]) 

        df.loc[start:end, ["action", "action_id"]] = ["", 0]

        df.to_csv(csv_file, index=False)
        print(f" Modificato {csv_file}: {start}-{end}")

# === MAIN ===
if __name__ == "__main__":
    xls = pd.ExcelFile(EXCEL_PATH)
    for sheet in SHEETS:
        df_corr = pd.read_excel(xls, sheet_name=sheet)
        print(f"\n=== Sheet {sheet} ===")
        apply_corrections(sheet, df_corr)
