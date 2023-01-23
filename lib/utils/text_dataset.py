from torch.utils.data import IterableDataset
import pdb
class TextDataset(IterableDataset):
    
    def __init__(self,input_file,column_index,max_char) -> None:
        super().__init__()
        self.file_name = input_file        
        self.text_col = column_index-1
        self.id_col = abs(column_index-2)
        self.max_char = max_char
        
    def preprocess(self,line):
        data = line.strip("\n ").split('\t')
        text_data = data[self.text_col]
        text_id = data[self.id_col]
        return text_id,text_data[0:self.max_char]
        
    
    def __iter__(self):
        f_itr = open(self.file_name,"r",encoding="utf-8")
        p_itr = map(self.preprocess,f_itr)
        return p_itr



