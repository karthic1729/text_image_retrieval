import faiss
import numpy as np
import os
import glob


class FaissANNImage:
    
    gpu_res = faiss.StandardGpuResources()
    output_sep1 = ","
    output_sep2 = "###"
    def __init__(self,index_path,shards_count,good_candidate_file) -> None:
        """
        Initializes with index shards loc & index_image_map loc
        """
        assert os.path.exists(index_path)
        self.index_path = index_path
        self.shards_count = shards_count
        index_shards = glob.glob(self.index_path+'/*.index')
        assert len(index_shards) == self.shards_count
        self.index_files = []
        self.index_image_files= []
        for index_id in range(1,self.shards_count+1):
            index_file = os.path.join(self.index_path, str(index_id)+'.index')
            index_image_map = os.path.join(self.index_path, 'CLIPImageData'+str(index_id)+'.tsv')
            assert os.path.exists(index_file) and os.path.exists(index_image_map) #do we need this
            self.index_files.append(index_file)
            self.index_image_files.append(index_image_map)
        self.valid_images = (img_id.strip("\n") for img_id in open(good_candidate_file,'r',encoding='utf-8'))


    def load(self,index_id):
        """
        Loads the index shard[index_id] to GPU 
        """
        self.index_id = index_id
        self.index_ivf = faiss.read_index(self.index_files[index_id])
        self.gpu_index_ivf = faiss.index_cpu_to_gpu(FaissANNImage.gpu_res, 0, self.index_ivf)
        self.index_image_map = [line.strip("\n ").split('\t')[0] for line in open(self.index_image_files[index_id],'r',encoding='utf-8')]

    def search_index(self,data,k,output_file):
        """
        finds k nearest images from all index shards
        k: nearest neighbour count
        data: text embeddings in h5py format(hardcoded and shared between text encoder & ann)
        """
        for index_id in range(0,self.shards_count):
            self.load(index_id)
            for batch_id in data.keys():
                clip_text_features = data[batch_id]["text_features"][:]
                input_data = data[batch_id]["input_ids"][:]
                D, I = self.gpu_index_ivf.search(clip_text_features, k)
                retrieved_images = []
                for inp_idx,text_data in input_data:
                    num_neighbors = len(D[inp_idx,:])
                    retrieval_results = [self.index_image_map[I[inp_idx,n_idx]]+FaissANNImage.output_sep2+str(D[inp_idx,n_idx]) for n_idx in range(num_neighbors) if I[inp_idx,n_idx]!=-1 and self.index_image_map[I[inp_idx,n_idx]] in self.valid_images]
                    if len(retrieval_results)>0:
                        retrieved_images.append(text_data+'\t'+FaissANNImage.output_sep1.join(retrieval_results))
                output_file.writelines(retrieved_images)