import torch 
import os,sys 
import torch.nn.functional as F 
import random 
import numpy as np 
import base64 
import time 
import json
import hashlib
import traceback
import math
import time
import random
import logging
from .model import MODEL 
import folder_paths
folder_paths.folder_names_and_paths["HMBS"] = ([os.path.join(folder_paths.models_dir, "HMBS_sty")], folder_paths.supported_pt_extensions)
loc_dir=os.path.dirname(os.path.abspath(__file__))

class PDF_STYLE:
    def __init__(self):
        vit_cfg={}
        vit_cfg["norm"]="ln"
        vit_cfg["num_head"]=16
        vit_cfg["num_trans_layers"]=[1,]
        vit_cfg["d_model"]=512
        vit_cfg["ch_in"]=512
        vit_cfg["ch_out"]=256
        vit_cfg["conv_blks"]=2
        vit_cfg["hidden_size"]=4096
        self.model=MODEL(**vit_cfg).cuda().bfloat16()
        self.model.load_state_dict(torch.load(os.path.join(loc_dir,"嘿嘿嘿.pth") ))
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sty_nm": (folder_paths.get_filename_list("HMBS"), ),
            },
        }
    RETURN_TYPES = ("CONDITIONING",)
    # RETURN_TYPES = ("IMAGE",)
    FUNCTION = "LOAD_STY"
    CATEGORY = "HMBS"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def LOAD_STY(self,sty_nm):
        sty_path=os.path.join(folder_paths.models_dir, "HMBS_sty",sty_nm)
        # dummy_input=torch.load("/home/offl/Documents/comfyui/ComfyUI/models/HMBS_sty/佳丽/冷白/0ea6454ab72ec39ae50ed03068165f08/a525.pth").bfloat16().cuda()
        dummy_input=torch.load(sty_path).bfloat16().cuda()
        
        cond=self.model(dummy_input)
        # cond=torch.load("/media/offl/系统/pjt_slow/ppt_test/gened_img/2025-05-23-14-17-27-2068/prompt_embeds.pth").bfloat16().cuda()
        bbox_out_tmp=torch.zeros(1,256,256,3)

        output={'pooled_output':torch.zeros(1,768)}
        return ([[cond, output]], )


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "PDF_STYLE":PDF_STYLE,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PDF_STYLE": "PDF_STYLE"
}



if __name__=="__main__":
    pass

