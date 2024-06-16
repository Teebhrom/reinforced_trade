import pickle
from finrl.meta.data_processor import DataProcessor

#split data_train into  chunks
with open('data_train.pkl', 'rb') as f:
    data = pickle.load(f)
    print(len(data))
    with open('price_array_train.pkl', 'rb') as f:
        price_array = pickle.load(f)
    with open('tech_array_train.pkl', 'rb') as f:
        tech_array = pickle.load(f)
    with open('turbulence_array_train.pkl', 'rb') as f:
        turbulence_array = pickle.load(f)
    data_chunks = [data[i:i + len(data) // 6] for i in range(0, len(data), len(data) // 6)]
    price_array_chunks = [price_array[i:i + len(price_array) // 6] for i in range(0, len(price_array), len(price_array) // 6)]
    tech_array_chunks = [tech_array[i:i + len(tech_array) // 6] for i in range(0, len(tech_array), len(tech_array) // 6)]   
    turbulence_array_chunks = [turbulence_array[i:i + len(turbulence_array) // 6] for i in range(0, len(turbulence_array), len(turbulence_array) // 6)] 
    
    for i, (chunk, price_array, tech_array, turbulence_array) in enumerate(zip(data_chunks, price_array_chunks, tech_array_chunks, turbulence_array_chunks)):
        dp = DataProcessor(data_source = 'yahoofinance')
        with open(f'data_train_{i}.pkl', 'wb') as f:
            pickle.dump(chunk, f)
        with open(f'price_array_train_{i}.pkl', 'wb') as f:
            pickle.dump(price_array, f)
        with open(f'tech_array_train_{i}.pkl', 'wb') as f:
            pickle.dump(tech_array, f)
        with open(f'turbulence_array_train_{i}.pkl', 'wb') as f:
            pickle.dump(turbulence_array, f)