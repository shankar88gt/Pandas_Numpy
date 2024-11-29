# Numpy Key concepts
"""
Numpy array's 
    all elements of same datatype
    np.array()
        np.zeros((3,2)) - zeros
        np.arange((3,2)) - continouts integers
        np.random.random(()) -  floats

    array.shape
    array.flatten() - converts into one dimension
    array.reshape(())  - must be compatible with the reshape size

    Numpy data types - .dtype
        int   - int8,int16,int32,int64
        float -   ----"------
        String - assumes longest string length 

        np.dtype as argument while creating 
        array.astype()

        type coercion - number to string - automatic
                      - int to float
                      - boolean to int 
        
    Numpy is also o indexed

    Indexing
        arr[2,4] - one element
        arr[0] - entire 0 row
        arr[:,3] - entire column
    slicing 1D array
        arr[2:4] - 4 not included
    slicing 2D array
        arr[3:6,3:6]  - row - 3 to 5, column 3 to 5
        slicing with steps
        arr[3:6:2, 3:6:2] 

    sorting
        np.sort(arr) - sort by columns ( horizontal sorting )
        axis 0 - row
        axis 1 - columns

    Filter
        Boolean array ( select only True )
        np.where(condition, when true, else) - returns array of indices

    Np.concatenate
        default along axis = 0
        matrix compatibility 
        works only existing array i.e. cannot convert 2D to 3D
        use reshape before concatenate

    np.delete(data, 1,axis=0) - delete second row

    Aggregating functions
        .sum()
            axis = 0 = column totals
            axis = 1 = row totals
        .min()
        .max()
        .mean()
        .cumsum()
            keepdims=True - the dimentions are collapsed

    Vectorization
        When NumPy sums elements in an array, it does not add each element one by one, but rather adds all of them together at once.
        NumPy is able to outsource tasks to C, a low-level programming language known for its speed
        Delegating tasks to C is a big reason for NumPy's own efficiency!  to perform a task can be anywhere from ten to a hundred times faster

    Broadcasting
            It's also possible to perform mathematical operations between arrays of different shapes. Since this involves "broadcasting", 
                or stretching the smaller array across the larger one, the conventions that govern this type of array math are called "broadcasting."

        Broading casting a scalar
        Compatibility Rules appply here
            (10,5) (10,1)  -   10 & 10 matches & 5 & 1 is ok or 5 & 5 is ok hence compatible
        
    Saving & Loading array
        .csv, .txt, .pkl , .npy ( best )

        with open("","rb") as f:
            fie = np.load(f)
            or
            np.save(f,array)
        
    help(function name)
    help(np.ndarray.flatten)

    np.flip(, axes=())
    np.transpose(,axes=())

    Stacking & splitting
        Stacking - to add a new dimension
        
        





    

    
    
        






    
    





"""

