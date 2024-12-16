# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:02:25 2024

@author: merye
"""

import pandas as pd

# Excel dosyasını yükle
file_path = r"C:\Users\merye\OneDrive\Masaüstü\Tez1\0.63VRHE_data_after_clearning.xlsx"
data = pd.read_excel(file_path, index_col=0)

# Formül oluşturmak için element sütunları
element_columns = ['Mn', 'Ni', 'Mg', 'Ca', 'Fe', 'La', 'Y', 'In']

# Formül oluşturma fonksiyonu (element adı ve değeri birleşik olacak şekilde)
def create_formula(row):
    formula_parts = [f"{element}{row[element]}" for element in element_columns if row[element] > 0]
    return ''.join(formula_parts)

# Her satır için formül oluşturun ve son sütun olarak ekleyin
data['formula'] = data.apply(create_formula, axis=1)

# Yeni verileri kaydetmek için dosya yolunu belirleyin
output_file_path = r"C:\Users\merye\OneDrive\Masaüstü\Tez1\0.63VRHE_data_after_formulas_updated.xlsx"
data.to_excel(output_file_path, index=False)

print(f"Formüller başarıyla eklendi. Yeni dosya: {output_file_path}")
