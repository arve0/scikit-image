import dask.array as da

__all__ = ['process_chunks']


def process_chunks(function, array, chunks=None, depth=None, mode=None):
    """
    function : function
        Function to be mapped which takes an array as an argument.
    array : numpy array
        array which the function will be applied to.
    chunks : tuple or tuple of tuples
        One tuple of length array.ndim or a list of tuples of length ndim.
        Where each subtuple adds to the size of the array in the corresponding
        dimension.
    depth : int
        integer equal to the depth of the internal external padding
    mode : 'reflect', 'periodic', 'wrap', 'nearest'
        type of external boundary padding

    Notes
    -----
    Be careful choosing the depth so that it is never larger than the length of
    a chunk.
    """
    if chunks is None:
        raise ValueError("`chunk` key word argument must be provided.")

    if mode == 'wrap':
        mode = 'periodic'

    darr = da.from_array(array, chunks=chunks)
    return darr.map_overlap(function, depth, boundary=mode)
