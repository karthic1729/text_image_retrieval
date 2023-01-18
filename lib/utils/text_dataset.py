from torch.utils.data import IterableDataset

class TextDataset(IterableDataset):
    def __init__(self,input_file) -> None:
        super().__init__()
        self.file_name = input_file        

    def __iter__(self):
        itr = open(self.file_name,"r",encoding="utf-8")
        return itr



