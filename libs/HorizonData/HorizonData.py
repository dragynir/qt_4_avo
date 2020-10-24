import pandas as pd

class HorizonData():
    def __init__(self,filename,header=None,sep=' ',drop_columns=None):
        self.data = pd.read_csv(filename, header = None,sep = ' ').dropna(axis="columns").drop(columns=drop_columns)
    
    def change_columns(self,values):
        print(values)
        newdict={}
        self.data =self.data.rename(columns=values)
        self.data=self.data
        