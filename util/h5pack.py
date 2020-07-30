import h5py
import numpy as np

numeric_types = {int, float}
primitive_types = {int, float, str, bool, type(None), np.ndarray}  # Numpy array behaves like a primitive for most purposes
collection_types = {tuple, list, dict, set}
indexed_types = {tuple, list}
associative_types = {dict, set}

collection_type_strs = {x.__name__ for x in collection_types}


# For converting data_type metadata
str_type_map = {
    'int': int,
    'float': float,
    'str': str,
    'bool': bool,
    'NoneType': type(None),
    'tuple': tuple,
    'list': list,
    'dict': dict,
    'set': set,
    'ndarray': np.ndarray,
    'int8': np.int8,  # Yeah, manually coding all the supported Numpy types here... could use np.ScalarType
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'uint64': np.uint64,
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
}


def is_integer_type(x_type):
    if x_type == int or issubclass(x_type, np.integer):
        return True
    return False


def is_number_type(x_type):
    if x_type in numeric_types or issubclass(x_type, np.number):
        return True
    return False


def is_primitive_type(x_type):
    if x_type in primitive_types or is_number_type(x_type):
        return True
    return False


def is_collection_type(x_type):
    if x_type in collection_types:
        return True
    return False


def is_collection_str(x_str):
    if x_str in collection_type_strs:
        return True
    return False


def is_indexed_homogeneous(data):
    """Returns True for homogeneous, False for heterogeneous.
    TODO: Special case of ints and floats mixed -> homogeneous float
    An indexed collection with any None (including only made of Nones) is heterogeneous.
    ndarray behaves like collection for this purpose.
    """
    type0 = type(data[0])
    if type0 == type(None):
        return False
    for item in data:
        item_type = type(item)
        if item_type != type0 or item_type in collection_types or item_type == np.ndarray:
            return False
    return True


def is_dict_homogeneous(data):
    """Returns True for homogeneous, False for heterogeneous.
    ndarray behaves like collection for this purpose.
    """
    k0, v0 = next(iter(data.items()))
    ktype0 = type(k0)
    vtype0 = type(v0)
    if ktype0 in collection_types or ktype0 == np.ndarray or vtype0 in collection_types or vtype0 == np.ndarray:
        return False
    for k, v in data.items():
        ktype = type(k)
        vtype = type(v)
        if (ktype != ktype0 or ktype in collection_types or ktype == np.ndarray) or \
                (vtype != vtype0 or vtype in collection_types or vtype == np.ndarray):
            return False
    return True


def is_set_homogeneous(data):
    """Returns True for homogeneous, False for heterogeneous."""
    k0 = next(iter(data))
    ktype0 = type(k0)
    if ktype0 in collection_types:
        return False
    for k in data:
        if type(k) != ktype0:
            return False
    return True


def validate_inds(keys):
    """Validate that the string keys in a group's sub-items form a valid set of indexes for a list/tuple. Raise
    ValueError if it fails. Note: keys strs are dumb lex order.
    """
    inds = sorted(int(ind) for ind in keys)
    target_inds = list(range(len(inds)))
    if inds != target_inds:
        raise ValueError('Keys don''t make up valid indexes')


def clean_key(key):
    """Ensure key is either an int or else coerced to a string"""
    # TODO: Possibly more sophisticated handling of this. Right now, just coerce everything into a str (when unpacking, the type may allow going back)
    if is_integer_type(type(key)):
        return str(key)
    return str(key)


def write_attrs(ds, attrs):
    """Write dataset attributes dict, including special handling of 'type' attr."""
    for k, v in attrs.items():
        if k == 'data_type' or k == 'collection_type' or k == 'key_type':
            try:  # For Python types
                v = v.__name__
            except AttributeError:  # For Numpy types
                v = str(v)
        ds.attrs[k] = v


def write_primitive(group, name, data):
    """Note: No dataset chunk options (like compression) for scalar"""
    data_type = type(data)

    # Write dataset
    if data_type == str:
        ds = group.create_dataset(name, data=np.string_(data))
    elif data_type == type(None):
        ds = group.create_dataset(name, data=0)
    else:
        ds = group.create_dataset(name, data=data)

    # Write attrs
    write_attrs(ds, {'data_type': data_type, 'collection_type': 'primitive'})
    return ds


def read_primitive(group, name):
    """"""
    ds = group[name]
    data_type = str_type_map[ds.attrs['data_type']]
    val = ds[...]
    if data_type == str:
        val = str(np.char.decode(val, 'utf-8'))
    elif data_type == bool:
        val = bool(val)
    elif data_type == type(None):
        val = None
    elif is_number_type(data_type):
        val = data_type(val)  # Convert back to scalar Python built-in or Numpy number type
    elif data_type == np.ndarray:
        pass  # Numpy type, keep as is
    else:
        raise ValueError('Scalar data type not recognized')
    return val


def write_indexed(group, name, data, ds_kwargs):
    """Write list or tuple"""
    data_type = type(data)
    homegeneous = is_indexed_homogeneous(data)
    type0 = type(data[0])

    if homegeneous:  # Save homogenous as numpy array
        item_type = type0
        if item_type == str:
            ds = group.create_dataset(name, data=np.string_(data), **ds_kwargs)
        else:
            ds = group.create_dataset(name, data=data, **ds_kwargs)
        write_attrs(ds, {'data_type': item_type, 'collection_type': data_type, 'homogeneous': True})
        return ds
    else:  # Save heterogeneous as a subgroup with indexed vals
        sub_group = group.create_group(name)
        for i, item in enumerate(data):
            write_data(sub_group, '{}'.format(i), item, ds_kwargs)
        write_attrs(sub_group, {'data_type': data_type, 'collection_type': data_type, 'homogeneous': False})
        return sub_group


def read_indexed(group, name):
    """Read list or tuple"""
    sub_group = group[name]  # A dataset for homogeneous; a group for heterogeneous
    collection_type = str_type_map[sub_group.attrs['collection_type']]
    homogeneous = bool(sub_group.attrs['homogeneous'])

    # Read homogeneous array as single val
    if homogeneous:
        ds = group[name]
        item_type = str_type_map[ds.attrs['data_type']]
        vals = ds[...]
        if item_type == str:
            vals = list(val.decode('utf-8') for val in vals)
        else:
            vals = list(item_type(val) for val in vals)
    else:
        keys = sub_group.keys()
        validate_inds(keys)
        vals = [None] * len(keys)
        for ind_str in sub_group.keys():
            ind = int(ind_str)
            vals[ind] = read_data(sub_group, ind_str)

    # Convert list to tuple if needed
    if collection_type == tuple:
        vals = tuple(vals)

    return vals


def write_associative(group, name, data, ds_kwargs):
    """Dicts (homogeneous and heterogeneous) are stored in a subgroup; Sets are stored like lists/tuples.
    Note: If heterogeneous, keys are packed as strings but restored to previous val on unpack.
    """
    data_type = type(data)

    if data_type == dict:
        # See if it's homogeneous - the keys are all 1 type and he vals are all 1 type
        homogeneous = is_dict_homogeneous(data)

        # Create subgroup that holds key/val datasets for homogeneous, or keys sub-items for heterogeneous
        sub_group = group.create_group(name)
        write_attrs(sub_group, {'data_type': data_type, 'collection_type': data_type, 'homogeneous': homogeneous})

        # Save homogeneous dict as 2 arrays
        if homogeneous:
            keys = []
            vals = []
            for k, v in sorted(data.items()):  # guaranteed to be orderable
                keys.append(k)
                vals.append(v)
            ktype = type(k)
            vtype = type(v)
            if ktype == str:
                keys = np.string_(keys)
            if vtype == str:
                vals = np.string_(vals)

            ds_keys = sub_group.create_dataset('keys', data=keys)
            ds_vals = sub_group.create_dataset('vals', data=vals)
            write_attrs(ds_keys, {'data_type': ktype})
            write_attrs(ds_vals, {'data_type': vtype})
        else:
            for k, v in data.items():
                ktype = type(k)
                k = clean_key(k)  # Turn key into string
                write_data(sub_group, k, v, ds_kwargs, key_type=ktype)  # add extra info for key type for unpacking

        return sub_group

    elif data_type == set:
        # Write like an indexed collection
        data_list = list(data)
        group_ = write_indexed(group, name, data_list, ds_kwargs)

        # Overwrite attrs
        homogeneous = is_set_homogeneous(data)
        if homogeneous:
            item_type = type(data_list[0])
        else:
            item_type = data_type
        write_attrs(group_, {'data_type': item_type, 'collection_type': data_type, 'homogeneous': homogeneous})
        return group_

    else:
        raise Exception('should not reach here')


def read_associative(group, name):
    """"""
    sub_group = group[name]
    collection_type = str_type_map[sub_group.attrs['collection_type']]
    homogeneous = bool(sub_group.attrs['homogeneous'])

    if collection_type == dict:
        if homogeneous:
            ds_keys = sub_group['keys']
            ktype = str_type_map[ds_keys.attrs['data_type']]
            keys = ds_keys[...]
            if ktype == str:
                keys = list(key.decode('utf-8') for key in keys)
            else:
                keys = list(ktype(key) for key in keys)

            ds_vals = sub_group['vals']
            vtype = str_type_map[ds_vals.attrs['data_type']]
            vals = ds_vals[...]
            if vtype == str:
                vals = list(val.decode('utf-8') for val in vals)
            else:
                vals = list(vtype(val) for val in vals)

            return {k: v for k, v in zip(keys, vals)}
        else:
            d = {}
            for key, key_group in sub_group.items():
                val = read_data(sub_group, key)
                ktype = str_type_map[key_group.attrs['key_type']]
                if ktype != str:  # Try to turn non-str key back into original type - should just be ints
                    key = ktype(key)
                d[key] = val
            return d

    elif collection_type == set:
        # Read like an indexed collection
        d = read_indexed(group, name)
        return set(d)

    else:
        raise ValueError('Associative type not recognized')


def write_collection(group, name, data, ds_kwargs):
    """"""
    data_type = type(data)

    # Check whether collection is indexed or associative
    if data_type in indexed_types:
        group_ = write_indexed(group, name, data, ds_kwargs)
    elif data_type in associative_types:
        group_ = write_associative(group, name, data, ds_kwargs)
    else:
        raise Exception('should not reach here')

    return group_


def read_collection(group, name):
    """"""
    collection_type = str_type_map[group[name].attrs['collection_type']]

    if collection_type in indexed_types:
        return read_indexed(group, name)
    elif collection_type in associative_types:
        return read_associative(group, name)
    else:
        raise Exception('Collection type not recognized')


def write_data(group, name, data, ds_kwargs, key_type=None):
    """Main data writing function, which is called recursively. Does the heavy lifting of determining the type and
    writing the data accordingly.
    Args:
        group: Previous group this will be attached to
        name: Name of current group or dataset to hold this data
        data: Data to store
        ds_kwargs: Options
        key_type: type for data arg when data is a key in a dict/set
    """
    data_type = type(data)

    # Check whether type is primitive or collection
    if is_primitive_type(data_type):
        group_ = write_primitive(group, name, data)
    elif is_collection_type(data_type):
        group_ = write_collection(group, name, data, ds_kwargs)
    else:
        raise ValueError('Data not one of the valid primitive or collection types')

    if key_type is not None:
        write_attrs(group_, {'key_type': key_type})

    return group_


def read_data(group, name):
    """"""
    collection_type_str = group[name].attrs['collection_type']
    data_type = str_type_map[group[name].attrs['data_type']]

    if is_collection_str(collection_type_str):
        return read_collection(group, name)
    elif is_primitive_type(data_type):
        return read_primitive(group, name)
    else:
        raise ValueError('Data type not recognized')


def pack(data, filename, compression=True):
    """Pack data into filename.
    Args:
        data: str, number (int or float), ndarray, or tuple, list, dict, set of them to save
        filename: str, name of file to save
        compression: bool, whether to gzip each dataset
    """
    # Setup dataset keyword args
    ds_kwargs = {}
    if compression:
        ds_kwargs['compression'] = 'gzip'

    # Open data file
    with h5py.File(filename, 'w') as f:
        # Recursively write out data
        write_data(f, 'root', data, ds_kwargs)


def unpack(filename):
    """Unpack data from filename"""
    with h5py.File(filename, 'r') as f:
        # Recursively build up read data
        data = read_data(f, 'root')
    return data
