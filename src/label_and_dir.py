import pandas as pd

def append_ext(fn):
    return fn+".jpg"


def label_and_dir():
    train_label = pd.read_csv(r'..\datasets\labels\traininglabel.csv',dtype=str)
    valid_label = pd.read_csv(r'..\datasets\labels\publictestlabel.csv',dtype=str)
    test_label = pd.read_csv(r'..\datasets\labels\privatetestlabel.csv',dtype=str)
    
    train_label["id"]=train_label["id"].apply(append_ext)
    valid_label["id"]=valid_label["id"].apply(append_ext)
    test_label["id"]=test_label["id"].apply(append_ext)
    
    # Define our example directories and files
    train_dir = r'..\datasets\Training'
    valid_dir = r'..\datasets\PublicTest'
    test_dir = r'..\datasets\PrivateTest'
    return train_dir, valid_dir, test_dir, train_label, valid_label, test_label


