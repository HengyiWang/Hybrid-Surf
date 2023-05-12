import torch
import torch.nn as nn
import numpy as np
import tinycudann as tcnn
from .hash_encoding import HashEmbedder, SHEncoder


def get_embedder(config, multires=10, i_embed=-1, n_features_per_level=2, n_levels=16):
    if i_embed == -1:
        return nn.Identity(), 3
    elif i_embed==0:
        if config.tcnn_encoding:
            embed = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                # "otype": "Frequency", 
                # "n_frequencies": 12
                "otype": "OneBlob", #Component type.
	            "n_bins": 16
                # "otype": "Identity"
                },
                dtype=torch.float
            )
            out_dim = embed.n_output_dims
        else:
            embed_kwargs = {
                        'include_input' : True,
                        'input_dims' : 3,
                        'max_freq_log2' : multires-1,
                        'num_freqs' : multires,
                        'log_sampling' : True,
                        'periodic_fns' : [torch.sin, torch.cos],
            }
            
            embedder_obj = Embedder(**embed_kwargs)
            embed = lambda x, eo=embedder_obj : eo.embed(x)
            out_dim = embedder_obj.out_dim
    
    elif i_embed==1:
        print('Use tcnn encoding')
        # TODO: tcnn_encoding should be modified
        if config.tcnn_encoding:
            bound = -1
            for box in config.bounding_box:
                if box.max() > bound:
                    bound = box.max()
            
            bound = bound.cpu().data.numpy()
            # per_level_scale = np.exp2(np.log2(config.finest_res * bound / n_levels) / (n_levels - 1))
            per_level_scale = np.exp2(np.log2(config.finest_res / n_levels) / (n_levels - 1))
            embed = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": config.log2_hashmap_size,
                "base_resolution": config.base_resolution,
                "per_level_scale": per_level_scale,
            },
            dtype=torch.float
        )
            out_dim = embed.n_output_dims
        
        else:
            embed = HashEmbedder(bounding_box=config.bounding_box, \
                            log2_hashmap_size=config.log2_hashmap_size, \
                            finest_resolution=config.finest_res, initial=config.init_grid)
        
            out_dim = embed.out_dim
    
    elif i_embed==2:
        print('Use SphericalHarmonics')
        if config.tcnn_encoding:
            embed = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
                },
                dtype=torch.float
            )
            out_dim = embed.n_output_dims

        else:
            embed = SHEncoder()
            out_dim = embed.out_dim
    
    return embed, out_dim

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)