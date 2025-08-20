import os
import sys
from typing import Union
import argparse

def count_bytes(path: str, diag: bool=False) -> int:
    """Tallies amount of bytes used locally by a file or folder.

    count_bytes returns how many bytes is used by a file or folder.  For
    a folder, count_bytes uses a loop with os.walk(path) to estimate how 
    much space is used.

    The os.walk(path) loop uses a tuple, i.e.,
        ```
        for root, dirs, files in os.walk(path):
            ...
        ```
    In each iteration, count_bytes removes elements of `dirs` containing 
    some string associated with cloud storage, including 'Google Drive', 
    'OneDrive', 'iCloud', and 'Dropbox'.  This function only attempts to 
    ignore cloud folders by checking `dirs`; if the cloud folder appears 
    elsewhere, such as the root or when it is input into os.walk(), then 
    count_bytes still iterates through its contents.

    Args:
        path:   input path to target file or folder
        diag:   display intermediate files found if set to True

    Returns:
        nbytes: disk space used by target in bytes
    """
    CLOUD_KEYWORD = {'OneDrive', 'Google Drive', 'iCloud', 'Dropbox'}
    nbytes = 0

    # Check if path can be found
    if not os.path.isdir(path) and not os.path.isfile(path):
        raise FileNotFoundError(f"Not found: {path}")
    
    # Return size if file
    if os.path.isfile(path):
        fnum = os.path.getsize(path)
        if diag:
            print(f"\t\t{fnum:>15} bytes \t({path})")
        return fnum # = nbytes
    
    # Iterate through contents of sub-folders
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not any(kw in d for kw in CLOUD_KEYWORD)]

        for jfile in files:
            fdir = os.path.join(root, jfile)
            fnum = os.path.getsize(fdir)
            if diag:
                print(f"\t\t{fnum:>15} bytes\t({fdir})")
            nbytes += fnum

    return nbytes

def format_byte_string(nbytes: int) -> str:
    """Converts bytes to a suitable format.
    
    format_byte_string converts some number of bytes into the units with 
    the smallest SI prefix possible such that there are at most 3 digits 
    preceding the decimal of the value rounded to two decimal places.
    
    This function converts the number of bytes into at least KB.  If the 
    number exceeds 1 PB, then this throws an error.

    Args:
        nbytes: number of bytes

    Returns:
        size:   string of space used in KB, MB, GB, or TB
        
    Examples:
        >>> format_byte_string(161803398874)
        '150.69 GB' # Value rounded to 2 decimals
        
        >>> format_byte_string(100.5 * 2**20)
        '100.5 MB'  # Trailing zeroes removed
        
        >>> format_byte_string(0)
        '0 KB'
    """
    prefix = ['KB', 'MB', 'GB', 'TB']
    order  = 0
    
    if nbytes >= 2**50:
        raise ValueError(f"Unexpectedly large file size: {nbytes}")
        
    nbytes /= 1024 # start with KB
    
    while nbytes >= 1024:
        order  += 1
        nbytes /= 1024
        
    return "{:.5g} {:s}".format(round(nbytes, 2), prefix[order])
    
def show_used_space(path: Union[str, list]=None, diag: bool=False):
    """Displays the disk space used by each element in a directory.
    
    show_used_space prints a table of each file or folder in a directory 
    alongside how much space it uses.  If the input path leads to a file
    instead, then this function prints out only that file size.  When no 
    path is input, and the OS is Windows, this checks both Program Files
    and Program Files (x86) directories within the C drive.

    Skips any folder associated with cloud storage, except cloud folders 
    whose paths are input to show_used_space.

    Args:
        path: input path or list of paths to target files and/or folders
        diag: display intermediate files found if set to True
    """
    MAX_LEN       = 30
    CLOUD_KEYWORD = {'OneDrive', 'Google Drive', 'iCloud', 'Dropbox'}
    used_space    = {}
    running       = 0
    
    if not path and sys.platform == 'win32': # default Program Files
        path = []
        if os.path.isdir('C:/Program Files'):
            path += ['C:/Program Files']
        if os.path.isdir('C:/Program Files (x86)'):
            path += ['C:/Program Files (x86)']

    if not path:
        print("usage: show_used_space(path, diag)")
        return
    elif isinstance(path, list):
        for ipath in path:
            show_used_space(ipath, diag)
        return

    print(path)

    if os.path.isfile(path):
        _, file = os.path.split(path)
        try:
            used_space[file] = count_bytes(path, diag)
            print(f"\t{format_byte_string(used_space[file])}")
        except OSError as err:
            print(f"\tWarning: Failed to access '{path}'. {err}")
        return

    for item_name in os.listdir(path):
        if any(kw in item_name for kw in CLOUD_KEYWORD):
            continue

        item_path = os.path.join(path, item_name)
        try:
            used_space[item_name] = count_bytes(item_path, diag)
            running += used_space[item_name]
            if len(item_name) > MAX_LEN:
                item_str = item_name[:MAX_LEN-3] + '...:'
            else:
                item_str = (item_name + ':').ljust(MAX_LEN)
            size_str = format_byte_string(used_space[item_name])
            running_str = format_byte_string(running)
            print(f"\t{item_str}\t{size_str.ljust(9)}\t({running_str})")
        except OSError as err:
            print(f"\tWarning: Failed to access '{item_path}'.  {err}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        "Shows how much space is used by each file or folder within a "
        "directory."
    ))
    parser.add_argument(
        "path",
        type=str,
        nargs='*',
        default='',
        help="Folder path(s) to check."
    )
    parser.add_argument(
        "-d", "--diag",
        type=bool,
        default=False,
        help="Set True to show sizes of discovered intermediate files."
    )
    args = parser.parse_args()
    
    show_used_space(args.path, args.diag)