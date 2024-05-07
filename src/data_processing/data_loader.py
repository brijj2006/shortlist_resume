import pandas as pd


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError as e:
        print('File not found. Please check the file path.')
        print('Exception:', e)
        exit()
    return data
