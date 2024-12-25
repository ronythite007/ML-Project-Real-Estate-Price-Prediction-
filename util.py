import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x1 = np.zeros(len(__data_columns))
    x1[0] = sqft
    x1[1] = bath
    x1[2] = bhk
    if loc_index>=0:
        x1[loc_index] = 1

    return round(__model.predict([x1])[0],2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations

    with open("C:/Users/Rohan/Desktop/ML_App/server/artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)["data_columns"]
        print(__data_columns)
        __locations = __data_columns[3:]

    global __model
    if __model is None:
        try:
            with open(r"C:/Users/Rohan/Desktop/ML_App/server/artifacts/banglore_home_prices_model.pickle", 'rb') as f:
                __model = pickle.load(f)
        except FileNotFoundError:
            print("Error: banglore_home_prices_model.pickle file not found.")
        except pickle.PickleError:
            print("Error: Failed to load pickle file.")
       
    print("loading saved artifacts...done")


def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))