import argparse
import h5py
import numpy as np
import torch
from encoder.text_encoder import TextEncoder
from torch.utils.data import DataLoader
from utils.text_dataset import TextDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='', type=str, required=True,help="The input file")
    parser.add_argument("--output_path", default = '', type=str, required = True,help = "Output filepath")
    parser.add_argument("--model_name",default = 'CLIP', type = str, required= False)
    parser.add_argument("--output_path_hdf5", default = '', type=str, required = True,help = "Output filepath")
    parser.add_argument("--batch_size", default = 2048, type=int, required = True,help = "Batch Size")
    parser.add_argument("--read_chunk_size", default = 32768,type=int, required = False)
    parser.add_argument("--column_index", default = 2, type=int, required = False)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = dict()
    config['data_path'] = args.data_path
    config['model_name'] = args.model_name
    config['batch_size'] = args.batch_size
    config['device'] = device
    
    model = TextEncoder(config)
    #input file should have 2 cols: id\ttext
    dataset = TextDataset(args.data_path)
    dataloader  = DataLoader(dataset, batch_size = args.read_chunk_size,pin_memory=True)
    batch_size = args.batch_size
    inp_col = args.column_index-1
    inp_hashid_col = abs(args.column_index-2)
    text_file_output = open(args.output_path,'w')
    feature_output_file = h5py.File(args.output_path_hdf5, 'w')
    for idx,chunk in enumerate(dataloader):
        sample_size = len(chunk)
        text_features = []
        inp_ids = []
        for bidx in range(0,sample_size, batch_size):
            batch_input = chunk[bidx*batch_size:bidx*batch_size+batch_size]
            text_data =list(map(lambda x: x.strip("\n ").split('\t')[inp_col],batch_input))
            text_ids = list(map(lambda x: x.strip("\n ").split('\t')[inp_hashid_col],batch_input))
            assert len(text_data) == len(text_ids)
            text_features.append(model.encode(text_data))
            inp_ids.append(text_ids)
            text_output = [text_ids[i]+'\t'+",".join(text_feature.astype('str'))+"\n" for i,text_feature in text_features]
            text_file_output.writelines(text_output)
        text_features = np.array(text_features,dtype='float32')
        grp = feature_output_file.create_group(idx)
        dset = grp.create_dataset("text_features", data=text_features)
        dset = grp.create_dataset("input_ids", data=inp_ids)
    feature_output_file.close()