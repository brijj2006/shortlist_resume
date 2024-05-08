import pandas as pd


# This script contains a function load_data that loads a CSV file containing resume data into a pandas DataFrame.
# It uses the pd.read_csv function from the pandas library to read the CSV file specified by file_path.
# If the file is not found, it raises a FileNotFoundError and prints an error message before exiting the program.
# Finally, it returns the loaded DataFrame data.
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError as e:
        print('File not found. Please check the file path.')
        print('Exception:', e)
        exit()
    return data
