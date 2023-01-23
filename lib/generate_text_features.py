from encoder.text_encoder import TextEncoder
from torch.utils.data import DataLoader
from utils.text_dataset import TextDataset
import numpy as np
import argparse
import h5py
import torch
import time
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='', type=str, required=True,help="The input file")
    parser.add_argument("--output_path", default = '', type=str, required = True,help = "Output filepath")
    parser.add_argument("--model_name",default = 'CLIP', type = str, required= False)
    parser.add_argument("--output_path_hdf5", default = '', type=str, required = True,help = "Output filepath")
    parser.add_argument("--batch_size", default = 2048, type=int, required = True,help = "Batch Size")
    parser.add_argument("--read_chunk_size", default = 5096,type=int, required = False)
    parser.add_argument("--column_index", default = 2, type=int, required = False)
    parser.add_argument("--max_char",default=512,type=int,required=False)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = dict()
    config['data_path'] = args.data_path
    config['model_name'] = args.model_name
    config['batch_size'] = args.batch_size
    config['device'] = device
    
    model = TextEncoder(config)
    #input file should have 2 cols: id\ttext
    dataset = TextDataset(args.data_path,args.column_index,args.max_char)
    dataloader  = DataLoader(dataset, batch_size = args.read_chunk_size,pin_memory=True)
    batch_size = args.batch_size
    text_file_output = open(args.output_path,'w')
    feature_output_file = h5py.File(args.output_path_hdf5, 'w')
    for idx,chunk in enumerate(dataloader):
        chunk_t1 = time.time()
        chunk_size = len(chunk[0])
        inp_ids = []
        text_output = []
        for bidx in range(0,chunk_size, batch_size):
            batch_t1 = time.time()
            text_ids = list(chunk[0][bidx:bidx+batch_size])
            text_data = list(chunk[1][bidx:bidx+batch_size])
            data_count = len(text_ids)
            batch_text_features = model.encode(text_data)
            if bidx==0:
                text_features = batch_text_features
            else:
                text_features = np.concatenate((text_features,batch_text_features),axis=0)
            inp_ids.extend(text_ids)
            text_output.extend([text_ids[i]+'\t'+",".join(batch_text_features[i].astype('str'))+"\n" for i in range(data_count)])
            batch_t2 = time.time()
            print(f'Batch time {batch_t2-batch_t1}')
            
        if len(text_output)>0:
            text_file_output.writelines(text_output)
            inp_ids = np.array(inp_ids,dtype="S64")
            grp = feature_output_file.create_group(str(idx))
            f_set = grp.create_dataset("text_features", data=text_features)
            id_set = grp.create_dataset("input_ids", data=inp_ids)
        chunk_t2 = time.time()
        print(f'Chunk time {chunk_t2-chunk_t1}')
    feature_output_file.close()
    text_file_output.close()