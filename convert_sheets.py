import pandas as pd

xl = pd.ExcelFile('dati_cacao.xlsx')
for sheet in xl.sheet_names:
    df = pd.read_excel('dati_cacao.xlsx', sheet_name=sheet)
    df.to_csv(f'dati_cacao_{sheet}.csv', index=False)
    print(f'Converted sheet {sheet} to dati_cacao_{sheet}.csv')