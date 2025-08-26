import os
import re
import numpy as np



def confirm_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def get_memory_usage():
    # Importing the library to measure RAM usage
    import psutil
    return psutil.virtual_memory()[2]


def replace_all_occurences_in_string(complete_string, old_word, new_word, identify_overlapping_occurences=False):
    """
    This function replaces all occurences of the {old_word} in {complete_string} by the {new_word}.
    
    Inputs:
        complete_string: the string in which words will be replaced.
        old_word: the string sequence that will be replaced.
        new_word: the string sequences that will replace the {old_word} sequence.
        identify_overlapping_occurences: if replacing the {old_word} seq with {new_word} seq 
            recreates the {old_word} seq and you need to replace it.
            
    Outputs:
        complete_string: new string similar to {complete_string} input, but {old_word} replaced with {new_word}.
    """
    
    if identify_overlapping_occurences:
        indices = [m.start() for m in re.finditer('(?={})'.format(old_word), complete_string)]
    else:
        indices = [m.start() for m in re.finditer(old_word, complete_string)]
    
    if len(indices) > 0:
        chunks_of_string = [complete_string[:indices[0]] + new_word]
        for i in range(1, len(indices)):
            chunks_of_string += [complete_string[indices[i-1]+len(old_word):indices[i]] + new_word]
        chunks_of_string += [complete_string[indices[-1]+len(old_word):]]
        complete_string = ''.join(chunks_of_string)
    
    return complete_string


def normalize(x: np.ndarray, normalization_standard: np.ndarray=None):
    min_ = np.min(x) if normalization_standard is None else np.min(normalization_standard)
    max_ = np.max(x) if normalization_standard is None else np.max(normalization_standard)
    return (x-min_)/(max_-min_) if max_>min_ else x


def de_normalize(x: np.ndarray, normalization_standard: np.ndarray=None):
    min_ = np.min(x) if normalization_standard is None else np.min(normalization_standard)
    max_ = np.max(x) if normalization_standard is None else np.max(normalization_standard)
    return x*(max_-min_)+min_ if max_>min_ else x


def exponential_normalize(x: np.ndarray, exponent: int=np.exp(1)):
    return (exponent**x)/np.sum(exponent**x)


def np_sigmoid(x: np.ndarray, z: int=1): return 1/(1+np.exp(-z*x))


def get_data(
    my_data, num_samples: int=100, 
    certain_class: int=None, 
    check_saved: bool=False, _type: str='test', 
    batch_size: int=64, shuffle: bool=False
) -> np.ndarray:
    
    _data = my_data.train if _type=='train' else my_data.test
    
    if certain_class is not None:
        load_file = f'__ignore__/{'train_' if _type=='train' else ''}data_samples_{certain_class}.npz'
        if os.path.exists(load_file) and check_saved:
            arr = np.load(load_file)
            x, y = arr['x'], arr['y']
        else:
            x, y = my_data.sample_data_of_certain_class(_data, batch_size=_data.__len__(), class_=certain_class)
            if check_saved:
                np.savez_compressed(load_file, x=x, y=y)
    else:
        x, y = my_data.sample_data(_data, num_samples=num_samples, batch_size=batch_size, shuffle=shuffle)
    x, y = x.detach().cpu().numpy(), y.numpy()
    
    return x[:num_samples], y[:num_samples]


