import pickle
import hickle as hkl 

def save_obj(path, file_fullname, obj): 
    file_name, file_type = file_fullname.split('.')
    file_path = path + file_fullname
    
    if file_type == 'pkl':
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    elif file_type == 'hkl':
        hkl.dump(obj, file_path, mode='w')
    elif file_type == 'txt':
        with open(file_path, 'w') as f:
            print(obj, file=f)
        

# Load data from file
def load_obj(path, file_fullname):
    file_name, file_type = file_fullname.split('.')
    file_path = path + file_fullname
    
    if file_type == 'pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_type == 'hkl':
        return hkl.load(file_path)

