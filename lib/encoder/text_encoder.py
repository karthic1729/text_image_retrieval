from multilingual_clip import pt_multilingual_clip
from multilingual_clip import Config_MCLIP
from .clip import load,tokenize
import torch
import transformers
import pdb

CLIP_MODEL = 'ViT-B/32'
MCLIP_TEXT_MODEL = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'

class MultilingualCLIP(transformers.PreTrainedModel):
    config_class = Config_MCLIP.MCLIPConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.transformer = transformers.AutoModel.from_pretrained(config.modelBase, cache_dir=kwargs.get("cache_dir"))
        self.LinearTransformation = torch.nn.Linear(in_features=config.transformerDimensions,out_features=config.numDims)

    @classmethod
    def _load_state_dict_into_model(cls, model, state_dict, pretrained_model_name_or_path, _fast_init=True):
        model.load_state_dict(state_dict)
        return model, [], [], []


class TextEncoder():

    def __init__(self,config:dict) -> None:
        self.device = config['device']
        self.model_name = config['model_name']
        if self.model_name == "CLIP":
            self.model, _ = load(CLIP_MODEL, device=self.device)
            self.transform = tokenize 
        elif self.model_name == "MCLIP":
            self.model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(MCLIP_TEXT_MODEL).to(self.device)
            self.transform = transformers.AutoTokenizer.from_pretrained(MCLIP_TEXT_MODEL)

    def encode_text_clip(self,text_data):
        """ 
        encode_text_clip - generates embeddings from clip text encoder
        text_data: input text data - (n x 1)
        text_features: text embeddings - (n x 512) 
        """
             
        tokenized_text = self.transform(text_data,truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokenized_text)
            text_features = text_features.cpu().numpy()
        return text_features

    def encode_text_mclip(self,text_data):
        """ 
        encode_text_mclip - generates embeddings from multilingual text encoder(XLM-RoBERTa)
        text_data: input text data - (n x 1)
        text_features: text embeddings - (n x 512) 
        
        Note: if run on GPU,model takes up 22GB GPU memory so batch size<=2048 due to memory limitation
        """
        with torch.no_grad():
            txt_tok = self.transform(text_data, padding=True, return_tensors='pt').to(self.device)
            embs = self.model.transformer(**txt_tok)[0]
            att = txt_tok['attention_mask']
            embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
            embs = self.model.LinearTransformation(embs).cpu().numpy()
        return embs

    def encode(self,text_data):
        """ 
        encode_text_mclip - generates text embeddings for clip & mlcip
        """ 
        if self.model_name == "CLIP":
            return self.encode_text_clip(text_data)
        elif self.model_name == "MCLIP":
            return self.encode_text_mclip(text_data)