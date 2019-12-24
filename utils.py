# Read and write files, the most robust way
def readfile(filepath):
	"""
	1. Use r or 'r+' if there are encoding and decoding error
    2. f.readline(): reads only the first line of the file.
    3. f.readlines(): reads all the lines and put them in list
    	and displays them with each line ending with'\n'
	"""
    with open(filepath, 'r') as f:
        data = f.read()
    return data



def write2file(content,filepath):
    with open(filepath, 'w') as f:
        f.write(content)


# Read and write json files
def save2json(data, filepath):
    with open(jsonfile_raw, "w") as ofile:
    # indent=4 to avoid writing json file in one line
        json.dump(data, ofile, indent =4)


def readjson(filepath):
	# use load() or loads()
    with open (filepath, encoding = 'utf-8') as f:
        data = json.load(f)
    return data



# Read and write pickle files
def dump_pickle(data, filename):
    with open(filename, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

	
def load_pickle(filename):
    with open(filename, 'rb') as pkl_file:
        filecontent = pickle.load(pkl_file)
    return filecontent


# Infinite Generator
def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


# Make directory
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# Update learning rate
def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr


# Save PyTorch Model checkpoint
def save_checkpoint(state, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)


# Load model from checkpoints
def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


