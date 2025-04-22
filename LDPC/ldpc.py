from email.quoprimime import decode

#from pyldpc.ldpc_images import decode_img

from . import code
from . import utils_img
from . import ldpc_images
from . import encoder
from . import decoder

class LDPC:
    def __init__(self, n, d_v, d_c, snr, seed = 0, systematic=True, sparse=True):
        self.n = n
        self.d_v = d_v
        self.d_c = d_c
        self.seed = seed
        self.systematic = systematic
        self.sparse = sparse
        self.H = None
        self.G = None
        self.bin_shape = None
        self.snr = snr

    def make_ldpc(self):
        H, G = code.make_ldpc(self.n, self.d_v, self.d_c, seed = self.seed, systematic=True, sparse=True)
        self.H = H
        self.G = G
        return H, G

    def set_snr(self, snr_val):
        self.snr = snr_val

    def get_snr(self):
        return self.snr

    def encode(self, data, is_image = True):
        if is_image:
            data = utils_img.rgb2bin(data)
            self.bin_shape = data.shape
            data_encoded = ldpc_images.encode_img(self.G, data, seed=self.seed)
        else:
            self.bin_shape = data.shape
            data_encoded = encoder.encode(self.G, data, seed=self.seed)
        return data_encoded

    def decode(self, img_encoded, max_iter, is_image = True):
        if is_image:
            img_decoded = ldpc_images.decode_img(self.G, self.H, img_encoded, self.bin_shape, self.snr, max_iter)
            return img_decoded
        else:
            data_decoded, reach = decoder.decode(self.H, img_encoded, self.snr, maxiter = max_iter)
            return data_decoded, reach


