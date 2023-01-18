from ann.faiss_ann import FaissANNImage
import os
import argparse
import h5py
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel",action="store true") #not handled right now due to multiple index shards
    parser.add_argument("--input_hdf5_path", default='', type=str, required=True, help="The input file")
    parser.add_argument("--output", default = '', type=str, required = True, help = "Output file")
    parser.add_argument("--index_path",default = '', type = str, required= True,help="index location in cluster")
    parser.add_argument("--valid_candidates_file",default='',type=str,required=True, help="file with all valid images")
    parser.add_argument('--k',default=20,type=int,required=False,help='k nearest neighours')
    args = parser.parse_args()

    assert os.path.exists(args.valid_candidates_file)
    assert os.path.exists(args.input_hdf5_path)    
    #reads the output of text_encoder
    input_data = h5py.File(args.input_hdf5_path, 'r')
    faiss_ivf_index = FaissANNImage(args.index_path,12,args.valid_candidates_file)
    output_file = open(args.output, 'w', encoding = 'utf-8')
    faiss_ivf_index.search_index(input_data,args.k,output_file)
    output_file.close()
    