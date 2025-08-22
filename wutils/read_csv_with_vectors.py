import pandas as pd
import numpy as np
import re
import os

def read_csv_with_vectors(ffile, **kwargs):
    """
    Runs pandas read_csv() with additional parsing for numpy arrays.

    Args:
        ffile (str): Path to CSV file
        **kwargs: Additional arguments to pass into read_csv()

    Returns:
        out (pandas.core.frame.DataFrame): Output from read_csv()
    """

    # Read file
    df = pd.read_csv(ffile, **kwargs)

    # Locate column variables stored as strings of vectors or matrices
    num_expr = r'[+-]?\d+\.?\d*'
    for key in df.keys():
        sample = df[key][0]  # assumes entries of column use same format
        
        if type(sample) is not str:
            continue

        nums = re.findall(num_expr, sample)

        if nums and '/' not in sample and '\\' not in sample:
            df[key] = df[key].apply(lambda x: np.array(atomize_vector_str(x)))

    return df

def atomize_vector_str(text):
    """Returns nested list of numbers (or list of numbers)"""
    components = find_outer_nest(text)

    if not re.findall(r'[+-]?\d+\.?\d*', text):  # not a number/vector
        return text
    
    if len(components) == 1:
        is_vector, delim = is_uni_dim(components[0])
        components[0] = components[0].strip()
        if delim == ')':
            components[0] = components[0].replace('),', ');')
            components[0] = components[0].replace(') ', ');')
            delim = ';'
        elif delim == ']':
            components[0] = components[0].replace('],', '];')
            components[0] = components[0].replace('] ', '];')
            delim = ';'
        if not is_vector:
            components = components[0].split(delim)

    if len(components) == 1 and is_vector:
        new_str = components[0].translate(str.maketrans('', '', '[]()'))
        return [float(x) for x in new_str.replace(',', ' ').split()]
    else:
        out = []
        for ii in components:
            out.append(atomize_vector_str(ii))
        if len(out) == 1:
            return out[0]
        else:
            return out

def find_outer_nest(text: str, start_char='[', end_char=']'):
    """Returns the outermost nest of text grouped within a string"""

    # Initialize
    out = []
    counter = 0
    start_index = -1

    # Loop through string
    for ii, char in enumerate(text):

        # Start grouping text
        if (char == start_char) and (counter == 0):  # begin group
            start_index = ii+1

        # Track nest level
        if (char == start_char):  # increment nest level
            counter += 1
        elif (char == end_char):  # decrement nest level
            counter -= 1

        # End grouping text
        if (char == end_char) and (counter == 0):  # record group
            out.append(text[start_index : ii])  # remove delimiters

        # Error check
        if (counter < 0):
            print('Reached negative nest level')
            raise ValueError
        
    # Return input if no group found
    if not out:
        out.append(text)

    return out
    
def is_uni_dim(text) -> tuple:
    """Check if a string represents a uni-dimensional vector"""

    # Ensure no mismatch in number of brackets
    if text.count('(') != text.count(')'):
        print(f'Mismatch in number of parentheses: {text}')
        raise ValueError
    elif text.count('[') != text.count(']'):
        print(f'Mismatch in number of brackets: {text}')
        raise ValueError
    
    # If there are brackets, then just check number of brackets
        # Note: [[1, 2]] is treated as multi-dimensional
    if '(' in text:
        return text.count('(') == 1, ')'
    elif '[' in text:
        return text.count('[') == 1, ']'
    # ... else, text may be something like '1, 2, 3; 4, 5, 6'

    # Get number of valid numbers in string
    nums  = re.findall(r'[+-]?\d+\.?\d*', text)
    numel = len(nums)

    # Define helper function to check if valid multi-dimensional
    def _is_valid_multi_dim(split_text):
        if len(split_text) in [1, numel]:  # is or can be uni-dimensional
            return False

        numel_split = []
        for ii in split_text:
            nums_split = re.findall(r'[+-]?\d+\.?\d*', ii)
            numel_split.append(len(nums_split))
        
        return (numel == sum(numel_split)) and len(set(numel_split)) == 1

    # Determine if string can represent a multi-dimensional vector...
    # ... when split by (',', ';', ' ', '\n')
    for delim in (',', ';', ' ', '\n'):
        if _is_valid_multi_dim(text.split(delim)):
            return False, delim

    return True, ''

def _test_read_csv_with_vectors():
    
    # Create sample CSV file
    test_file = 'test_file.csv'
    if os.path.exists(test_file):
        print(f'{test_file} already exists.  Remove before running unit test.')
        raise FileExistsError
    test_data = {
        'timestamps': ['1234567890', '1.23456e9', '9876543210'],
        'filepaths': ['/path/to/file1.jpeg', 'path/to/file2.png', 'file3.tiff'],
        'vectors_space': ['1 2 3', '4 5 6', '7 8 9'],
        'vectors_comma': ['1, 2, 3', '4, 5, 6', '7, 8, 9'],
        'vectors_brackets': ['[1 2 3]', '[4, 5, 6]', '[[7 8 9]]'],
        'mixed': ['42', 'hello world', '[1, 2, 3]'],
        'matrices': [
            "[1 2 3; 4 5 6]",
            '[[-1, -2, -3], [-4, -5, -6]]',
            "10 20 30; 40 50 60"
        ]
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(test_file, index=False, quoting=2)

    # Define truth set
    truth_df = test_df
    truth_df.loc[0:3, 'timestamps'] = [np.array(1234567890), np.array(1.23456e9), np.array(9876543210)]
    truth_df.loc[0:3, 'vectors_space'] = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    truth_df.loc[0:3, 'vectors_comma'] = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    truth_df.loc[0:3, 'vectors_brackets'] = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    truth_df.loc[0:3, 'mixed'] = [np.array(42), 'hello world', np.array([1, 2, 3])]
    truth_df.loc[0:3, 'matrices'] = [
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[-1, -2, -3], [-4, -5, -6]]),
            np.array([[10, 20, 30], [40, 50, 60]])
    ]

    # Load data
    load_df = read_csv_with_vectors(test_file)

    # Delete test file
    os.remove(test_file)
    if os.path.exists(test_file):
        print(f'Failed to delete {test_file} during unit test.')
        raise FileExistsError

    # Run checks
    if not all(load_df.keys() == truth_df.keys()):
        return False
    
    for key in truth_df.keys():
        for ii, value in enumerate(truth_df[key]):
            check = np.bool(value == load_df[key][ii])
            if (check.size > 1) and not check.all():
                return False
            elif (check.size == 1) and not check:
                return False

    return True

if __name__ == "__main__":

    # Perform unit test
    if _test_read_csv_with_vectors():
        print("Unit test for 'read_csv_with_vectors' passed.")
    else:
        print("Unit test for 'read_csv_with_vectors' failed.")