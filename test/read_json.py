import json


with open(r'E:\시연\BG\spec_data_filler_BG.json', 'r', encoding = 'utf8') as f:
    data = json.load(f)
    
print(data.keys())
print(data['meta'])
print(data['classes'])



print(data['spectra'][0])
# print(len(data['spectra']))

