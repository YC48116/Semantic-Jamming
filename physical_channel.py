import subprocess
from idlelib.outwin import file_line_pats

import matplotlib.pyplot as plt
import numpy as np
# from paddle.fluid.libpaddle.eager.ops.legacy import label_smooth

# import torch
# from holoviews.plotting.mpl import BasicGridPlot
# from tensorflow.python.framework.test_ops import none_eager_fallback
import Modulaion_methods
from LDPC import ldpc
from PIL import Image
from time import time

##################################################################
# Let's see the image we are going to be working with
'''
n = 200
d_v = 15
d_c = 20


n = 12
d_v = 3
d_c = 6
seed = 42
is_image = False

ldpc = ldpc.LDPC(n = n, d_v = d_v, d_c = d_c, seed = seed)
H, G = ldpc.make_ldpc()
print(H.shape, G.shape)
'''
import BPG
import os

"""

Compression Part

"""
# total_psnr = 0.0


start = 10000
img_num = 1
psnrs = np.arange(5, 10, 0.5)
results = np.zeros_like(psnrs).astype(float)
idx = 0


class ImageTransmission:
    '''
        有两个补位置的情况：
        码长 200 个数 30 200*30
        1 码长不够LDPC码长 要补充
        2 QPSK调制位数不够需要补充
    '''
    # m = n * d_v / d_c
    # Rate = (n-m)/m
    # Rate = 1/2 n = 256 d_v = 63 d_c = 128
    # Rate = 3/4 n = 256 d_v = 31 d_c = 128
    # Rate = 7/8 n = 256 d_v = 15 d_c = 128
    def __init__(self, snr, rate = 5/6, mmn = "QPSK", n = 264 , d_v = 63, d_c = 132, seed=42, compress_rate = 1/6, is_image=False):

        self.snr = snr
        self.n = n
        # n = 100 d_v = 10 d_c = 20
        # n = 12 d_n = 3, d_v = 6
        # n = 200 d_n = 15, d_v = 20
        self.d_v = d_v
        self.d_c = d_c
        self.seed = seed
        self.is_image = is_image
        self.input_image = None
        self.test_bytes = None
        self.extend_len = 0
        self.is_append = False
        self.bin_data = None
        self.final_data = None
        self.data_to_decoded = None
        self.data_fft_r = None
        self.rate = rate
        self.file_path = None

        # Initialize LDPC
        if rate == 5/6:
            self.d_v = 43
        elif rate == 2/3:
            self.d_v = 87
        elif rate == 1/2:
            self.d_v = 131

        self.ldpc = ldpc.LDPC(n=self.n, d_v=self.d_v, d_c=self.d_c, snr = self.snr , seed=self.seed)
        print('dv dc', self.d_v, self.d_c)
        self.ldpc.set_snr(self.snr)
        self.H, self.G = self.ldpc.make_ldpc()

        self.inputBS = self.G.shape[0]
        self.compress_rate = compress_rate
        self.block_num = 0
        self.quality = 51
        self.module_method_name = mmn
        self.module = None
        self.de_module = None
        self.get_module()
        self.get_block_num()
        print('block_num: ', self.block_num)
        print(self.de_module, self.module)

    def get_module(self):
        if self.module_method_name == "QPSK":
            self.module = Modulaion_methods.qpsk_modulate
            self.de_module = Modulaion_methods.qpsk_demodulation
        elif self.module_method_name == "16-QAM":
            self.module = Modulaion_methods.qam16_modulate
            self.de_module = Modulaion_methods.qam16_demodulate
        elif self.module_method_name == "32-QAM":
            self.module =  Modulaion_methods.qam32_modulate
            self.de_module = Modulaion_methods.qam32_demodulate

    def get_block_num(self):

        if self.compress_rate == 1 / 6:
            self.block_num = 2
        elif self.compress_rate == 1 / 3:
            self.block_num = 2 * 2
        elif self.compress_rate == 1 / 2:
            self.block_num = 2 * 3
        elif self.compress_rate == 2 / 3:
            self.block_num = 2 * 4
        elif self.compress_rate == 5 / 6:
            self.block_num = 2 * 5
        else:
            self.block_num = 2 * 6

    def read(self, input_image):
        self.input_image = input_image

    def compress_encode_module(self, file_name):
        # print(file_name)
        if not self.file_path in file_name:
            file_name = self.file_path + str(file_name)  # 输入图片路径

        # print(file_name)
        self.read(file_name)
        #self.quality = 51
        output_bpg = f"./out_enc_{self.module_method_name}_{self.d_v}_{self.snr}_{1000*self.rate}.bpg" # 输出的BPG文件路
        # print(input_image)
        BPG.compress_to_bpg(self.quality, file_name, output_bpg)

        f = open(output_bpg, "rb")
        # print(os.stat(output_bpg))

        buf_len = os.stat(output_bpg)[6]
        buf = f.read()
        f.close()
        #print('After BPG size: ', buf_len, buf_len * 8)

        bin_data = BPG.bytes_to_bin(buf)
        # print('Compress Rate: ',len(bin_data)/(32*32*24))

        scale = 1
        if self.module_method_name == "16-QAM":
            scale = 4
        elif self.module_method_name == "32-QAM":
            scale = 5
        else:
            scale = 2

        if len(bin_data) < scale * self.block_num * self.G.shape[1]:
            output_bpg = f"./out_enc_{self.module_method_name}_{self.d_v}_{self.snr}_{1000*self.rate}.bpg"  # 输出的BPG文件路
            # print(input_image)
            quality = self.quality
            while quality > 0 and len(bin_data) < scale * self.block_num * self.G.shape[1]:
                quality = quality - 1
                BPG.compress_to_bpg(quality, file_name, output_bpg)

                f = open(output_bpg, "rb")
                # print(os.stat(output_bpg))

                buf_len = os.stat(output_bpg)[6]
                buf = f.read()
                f.close()
                #print('After BPG size: ', buf_len, buf_len * 8)

                bin_data = BPG.bytes_to_bin(buf)
                #print('try quality ',quality , len(bin_data) , scale * self.block_num * self.G.shape[1])

            if quality < 0:
                self.quality = 0
            else:
                self.quality = quality + 1
            # print('Compress Rate: ',len(bin_data)/(32*32*24))

        BPG.compress_to_bpg(self.quality, file_name, output_bpg)

        f = open(output_bpg, "rb")
        # print(os.stat(output_bpg))

        buf_len = os.stat(output_bpg)[6]
        buf = f.read()
        f.close()
        print('After BPG size: ', buf_len, buf_len * 8)

        bin_data = BPG.bytes_to_bin(buf)
        print('final quality: ', self.quality, scale * self.block_num * self.G.shape[1], len(bin_data))
        to_extend = scale * self.block_num * self.G.shape[1] - len(bin_data)
        bin_data = np.concatenate((bin_data, np.zeros(to_extend).astype(int)))

        self.extend_len = to_extend
        print('to_extend: ', self.extend_len, self.G.shape, self.H.shape)
        # print('extend_len: ',len(bin_data))
        assert (len(bin_data) % self.G.shape[1] == 0)
        self.quality = 51
        test_bytes = BPG.bin_to_bytes(bin_data)

        self.test_bytes = test_bytes
        # print(test_bytes == buf)
        is_append = False
        # print(bin_data.shape)

        self.is_append = is_append

        # print(len(bin_data))

        bin_data = bin_data.reshape(self.G.shape[1], -1)
        # print(bin_data)
        print('bin_data shape',bin_data.shape)

        self.bin_data = bin_data

        # print(type(bin_data))
        # tiger = np.asarray(Image.open('10000.jpg'))
        """

        LDPC Encode

        """
        # Compression
        # bin_data = torch.tensor(bin_data)
        img_encoded = self.ldpc.encode(bin_data, is_image=self.is_image)
        img_encoded = img_encoded.T
        # random_data = np.random.choice([-1, 1], size=(1000, 200, 384))
        # print('LDPC Encoded shape: ', img_encoded.shape)
        print('After encoding', img_encoded.shape)
        # print(np.max(img_encoded), np.min(img_encoded))
        # mapped_bits = np.where(img_encoded == -1, 0, 1)
        """

        QPSK Modify

        """

        qpsk_symbols = np.array([self.module(row) for row in img_encoded])

        return qpsk_symbols



    def channel(self, symbols, is_fading=False):
        # print('QPSK shape: ', qpsk_symbols.shape)

        '''
        #inputBS = img_encoded_qpsk.shape[0]
        #size = int(img_encoded_qpsk.shape[1]/2)
        #img_encoded = img_encoded.reshape(inputBS, size, 2)

        img_encoded_complex = img_encoded[...,0] + 1j*img_encoded[...,1]
        '''

        """

        OFDM & ifft

        """

        img_encoded_complex_ifft = np.fft.ifft(symbols, n=None, norm='ortho')
        # print('After ifft: ',img_encoded_complex_ifft.shape)
        #img_encoded_complex_ifft = self.norm_energy(img_encoded_complex_ifft)

        #print('img_encoded_complex_ifft: ',np.mean(np.sum(symbols ** 2, axis=1)))
        """

        AWGN Channel

        """
        # N = 48
        # snr = 5
        hstd = 1  # Rayleigh Fading
        # while(snr):
        # numOFDM = int(img_encoded_complex_ifft.shape[1] / N)
        # img_encoded_complex_ifft = img_encoded_complex_ifft.reshape(img_encoded_complex_ifft.shape[0], numOFDM, N)

        inputBS, len_data = img_encoded_complex_ifft.shape[0], img_encoded_complex_ifft.shape[1]
        self.inputBS = inputBS
        noise_std = 10 ** (-self.snr * 1.0 / 10 / 2)
        AWGN = np.random.normal(0, noise_std, size=img_encoded_complex_ifft.shape)

        hh = np.random.rayleigh(hstd, 1)
        # print('Fading: ', hh)
        # print(hh.shape, AWGN.shape,img_encoded_complex_ifft.shape)
        # data = hh * img_encoded_complex_ifft + AWGN
        if is_fading:
            data = hh * img_encoded_complex_ifft + AWGN
            return data

        data = img_encoded_complex_ifft + AWGN
        return data

    def decompress_decode_module(self, data, flag_rl=False):
        """

        AWGN Channel

        """

        if type(data) == (torch.Tensor):
            data = data.cpu()

        print(data.shape)

        data_fft = np.fft.fft(data, n=None, norm='ortho')
        print('After fft: ',data_fft.shape)
        """

         De QPSK

        """
        data_fft = np.array([self.de_module(row) for row in data_fft])
        data_fft = data_fft.T
        print('After qpsk_demodulation: ',data_fft.shape)
        # mapped_bits = np.where(data_fft == 0, -1, 1)

        # print(mapped_bits.shape)
        # print(data_fft.shape)
        # data_fft_r = np.concatenate([data_fft.real[..., np.newaxis], data_fft.imag[..., np.newaxis]], axis=-1)
        """

        FFT

        """

        # print(self.inputBS)
        # data_fft_r = data_fft.reshape(self.inputBS, -1)

        """

        LDPC Decode

        """
        # print(data_fft_r.shape)
        # print(data_fft_r == img_encoded)
        max_iter = 20

        self.data_fft_r = data_fft

        data_to_decoded, reach = self.ldpc.decode(self.data_fft_r, max_iter=max_iter, is_image=self.is_image)
        # \print(tiger_decoded[:8, :] == bin_data)

        # print(tiger_decoded.shape)

        self.data_to_decoded = data_to_decoded
        print(data_to_decoded.shape,)
        print('After LDPC decode：', data_to_decoded.shape)

        final_data = data_to_decoded[:self.G.shape[1], :]

        print('After LDPC decode：', final_data.shape)

        self.final_data = final_data

        # print(cal_SER(final_data, bin_data),cal_BLER(final_data, bin_data))

        final_data = final_data.reshape(-1)

        # print(np.sum(~comparison))
        # final_data = final_data[:-self.extend_len]
        final_data_bytes = BPG.bin_to_bytes(final_data)
        print('|'*2, len(final_data_bytes), len(self.test_bytes))
        final_data_real = BPG.bin_to_bytes(final_data[:-self.extend_len])

        nums = 0
        for i in range(len(self.test_bytes)):
            if self.test_bytes[i] != final_data_bytes[i]:
                nums += 1

        print(f'Number of different bits: {nums} / {len(final_data_bytes)}')
        # print('8'*2,final_data_real.shape)
        # print(final_data_bytes == self.test_bytes)
        import cal_psnr
        from skimage import io
        if flag_rl:
            # print(len(final_data_real))
            # Start to decode
            with open(f"./out_enc {self.module_method_name}.bpg", "wb") as f:
                f.write(final_data_real)
            f.close()

            flag = BPG.decompress_from_bpg(f"./out_enc {self.module_method_name}.bpg", f"./out_dec {self.module_method_name}.png")
            if flag:
                img1 = io.imread(self.input_image)
                print(img1.shape)
                img2 = io.imread(f"./out_dec {self.module_method_name}.png")
                height, width, channels = img1.shape
                # BPG header damage
                if img1.shape == img2.shape:
                    psnr = cal_psnr.cal_psnr(img1, img2)
                else:
                    img2 = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
                    psnr = cal_psnr.cal_psnr(img1, img2)
                print('psnr: ', psnr)
                return img2, psnr, not reach
            else:
                # print(self.input_image)
                img1 = io.imread(self.input_image)
                height, width, channels = img1.shape

                img2 = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
                psnr = cal_psnr.cal_psnr(img1, img2)
                # total_psnr += psnr
                print('psnr: ', psnr)
                return img2, psnr, False

        else:
                # covert bytes into BPG format

                """

                    Calculate PSNR

                """
                with open(f"./out_dec_{self.module_method_name}_{self.d_v}_{self.snr}_{1000*self.rate}.bpg", "wb") as f:
                    f.write(final_data_real)
                f.close()

                flag = BPG.decompress_from_bpg(f"./out_dec_{self.module_method_name}_{self.d_v}_{self.snr}_{1000*self.rate}.bpg",
                                               f"./out_dec_{self.module_method_name}_{self.d_v}_{self.snr}_{1000*self.rate}.png")
                #flag = True
                if flag:
                    print('success')
                    img1 = io.imread(self.input_image)
                    print('img1 shape',img1.shape)
                    height, width, channels = img1.shape
                    img2 = io.imread(f"./out_dec_{self.module_method_name}_{self.d_v}_{self.snr}_{1000*self.rate}.png")
                    if img1.shape == img2.shape:
                        psnr = cal_psnr.cal_psnr(img1, img2)
                    else:
                        ave_pixel_val = np.mean(img1)
                        img2 = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
                        psnr = cal_psnr.cal_psnr(img1, img2)
                    # total_psnr += psnr
                    print('psnr: ', psnr)
                    return img2, psnr, not reach
                else:
                    print('fails ')
                    # If could not covert into BPG format and use random noise as the final output image
                    print(self.input_image)
                    img1 = io.imread(self.input_image)
                    #print(np.max(img1), np.min(img1))
                    height, width, channels = img1.shape
                    ave_pixel_val = np.mean(img1)
                    img2 = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
                    psnr = cal_psnr.cal_psnr(img1, img2)
                    # total_psnr += psnr
                    print('psnr: ', psnr)
                    return img2, psnr, False

    def cal_SER(self, received_symbols, transmitted_symbols):
        x, y = received_symbols.shape
        symbol_errors = np.sum(transmitted_symbols != received_symbols)
        SER = 1.0 * symbol_errors / (x * y)
        return SER

    def cal_BLER(self, received_symbols, transmitted_symbols):
        block_errors = np.sum(np.any(received_symbols != transmitted_symbols, axis=1))
        bler = 1.0 * block_errors / len(received_symbols)
        return bler

    def cal_SER_BLER(self):

        ser = self.cal_SER(self.bin_data, self.final_data)
        bler = self.cal_BLER(self.bin_data, self.final_data)

        return ser, bler

    def after_channel(self, file):
        self.read(self.file_path + str(file))
        data_before = self.compress_encode_module(self.file_path  + str(file))
        data_channel = self.channel(data_before, is_fading=False)
        return data_channel

    import tqdm

    def run(self, files):
        total_psnr = 0.0
        for f in tqdm.tqdm(files):
            print(f)
            data_before = self.compress_encode_module(self.file_path  + str(f))
            print(data_before.shape)

            data_channel = self.channel(data_before, is_fading=False)
            print('After channel', data_channel.shape)

            # 提取实部和虚部
            real_part = np.real(data_channel)  # 实部，形状 (200, 16)
            imag_part = np.imag(data_channel)  # 虚部，形状 (200, 16)

            data_insert = real_part + 1j * imag_part

            _, psnr,_ = self.decompress_decode_module(data_channel, flag_rl=False)
            ser, bler = self.cal_SER_BLER()
            print(f"{ser * 100:.2f} % {bler * 100:.2f} %")
            total_psnr += psnr
            # print(psnr)

            print()
        return 1.0 * total_psnr / len(files)


from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import tqdm


class Cifar10_dataset(Dataset):
    def __init__(self, files, snr=10, image_folder='./data/png/', transform=None):
        """
        参数：
        - files: 图像文件列表
        - image_folder: 存储图片的文件夹路径
        - transform: 可选的转换（如标准化、数据增强等）
        """
        self.files = files
        self.image_folder = image_folder
        self.transform = transform
        self.snr = snr
        self.img_real, self.img_imag, self.img_labels, self.file_names = self.generate_dataset(files, snr=self.snr)

    def set_snr(self, snr):
        self.snr = snr

    def generate_dataset(self, files, snr):
        img_real = []
        img_imag = []

        img_labels = []
        img_file_name = []
        IT = ImageTransmission(snr=self.snr)
        files_len = len(files)
        cnt = 0
        for filename in tqdm.tqdm(files):
            # 从文件名中提取标签（假设格式是数字_标签.png）
            label = int(filename.split('_')[1].split('.')[0])
            # print(filename)
            # 加载图像
            # img = Image.open(os.path.join(self.image_folder, filename))

            img = IT.after_channel(filename)

            if self.transform is not None:
                img = self.transform(img)

            # 提取实部和虚部
            real_part = np.real(img)  # 实部，形状 (200, 16)
            imag_part = np.imag(img)  # 虚部，形状 (200, 16)

            # 将图像和标签分别添加到列表
            img_real.append(real_part)
            img_imag.append(imag_part)
            img_labels.append(label)
            img_file_name.append(filename)
            cnt = cnt + 1
            print(cnt, '/', files_len, ' ends')
        # print(len(img_files), len(img_labels))
        return img_real, img_imag, img_labels, img_file_name

    def __len__(self):
        return len(self.img_imag)

    def __getitem__(self, idx):
        return self.img_real[idx], self.img_imag, self.img_labels[idx], self.file_names[idx]


import torch
from concurrent.futures import ThreadPoolExecutor

def process_snr(snr, png_files, rate, c_r, mmn):
    """处理单个 SNR 值的任务"""
    transmission = ImageTransmission(rate=rate, compress_rate=c_r, snr=snr, mmn=mmn)
    print(f"Processing SNR {snr} with method {transmission.module_method_name}")
    tp = transmission.run(png_files)
    print('_' * 10)
    print(snr, tp)
    print('_' * 10)
    return tp

def main():
    file_path = './data/png/'
    png_files = [f for f in os.listdir(file_path)]
    png_files = png_files
    snr = 0
    snrs = np.arange(6, 7, 1)
    psnrs = []
    c_r = 1/6
    mmn_s = ['QPSK','16-QAM']
    mmn = "QPSK"
    rates = [1/2, 2/3, 5/6]
    rate = 5/6

    # 创建线程池，最多使用4个线程
    with ThreadPoolExecutor(max_workers = 1) as executor:
        # 提交任务，每个 snr 值都提交给线程池处理
        futures = [executor.submit(process_snr, snr_value, png_files, rate, c_r, mmn) for snr_value in snrs]

        # 等待所有线程完成并获取结果
        for future in futures:
            tp = future.result()  # 阻塞等待当前线程完成
            psnrs.append(tp)

    print(snrs)
    print(psnrs)
    print(rate, mmn)

if __name__ == "__main__":
    main()

    '''
    # print(tp)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((200, 20)),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 如果图像是 RGB 格式
    ])

    dataset = Cifar10_dataset(files=png_files, image_folder=file_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    print(dataloader.dataset)
    # dataset.set_snr(10)
    # 保存为 .pth 文件
    # dataloader = DataLoader(dataset, batch_size = len(dataset), shuffle=True)
    img_real = dataset.img_real
    img_imag = dataset.img_imag
    labels = dataset.img_labels
    names = dataset.file_names
    torch.save({'img_real': img_real, 'img_imag': img_imag, 'labels': labels, 'names': names}, f'batch_test.pth')
    # print(f"Batch {batch_idx} test saved.")

    data = torch.load('./batch_test.pth')
    # print(data)
    img_real = data['img_real']
    img_real = np.array(img_real)
    img_real = torch.tensor(img_real).float().squeeze(1)

    img_imag = data['img_imag']
    img_imag = np.array(img_imag)
    img_imag = torch.tensor(img_imag).float().squeeze(1)

    labels = data['labels']
    names = data['names']

    # 确保数据是 PyTorch 张量
    # images = torch.tensor(images)
    labels = torch.tensor(labels)

    # names = torch.tensor(names)
    print(type(img_real), type(labels), type(names))
    print(img_real.shape, img_imag.shape)

    # 创建一个足够大的空张量来存储交错的实部和虚部
    interleaved_data = torch.zeros((img_real.shape[0], img_real.shape[1], img_real.shape[2] * 2),
                                   dtype=torch.float32)  # (200, 32)
    print(interleaved_data.shape)

    # 将实部和虚部交替存储
    interleaved_data[..., ::2] = img_real  # 实部存储在偶数位置
    interleaved_data[..., 1::2] = img_imag  # 虚部存储在奇数位置

    restored_real_part = interleaved_data[..., ::2]  # 提取偶数位置的实部
    restored_imag_part = interleaved_data[..., 1::2]  # 提取奇数位置的虚部

    assert (restored_real_part.equal(img_real))
    assert (restored_imag_part.equal(img_imag))

    dataset = TensorDataset(interleaved_data, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for index, (images, labels) in enumerate(dataloader):
        print(images.shape, labels, names[index])

    print(len(dataset))


transmission = ImageTransmission(snr = 20)

#transmission.generate_dataset(png_files)
tp = transmission.run(png_files)
print(tp)

dataset = Cifar10_dataset(files = png_files, image_folder = file_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# 测试数据加载
print(dataset.__len__())
'''
#snrs = np.arange(8,15,1)
#psnrs = []
#for snr in snrs:
#transmission = ImageTransmission(snr = 6)

#transmission.generate_dataset(png_files)
#tp = transmission.run(png_files)
print("||||||||||")
#psnrs.append(tp)
#print(tp)
print("||||||||||")

#print(psnrs)
#print(tp)

#results[idx] = 1.0*(total_psnr/img_num)
#idx +=1

#print(total_psnr/img_num)
#print(results)
#snr += 1

#print(bin_data)

#print(tiger_decoded.shape, bin_data.shape)
'''


plt.imshow(tiger)
plt.show()

plt.imshow(tiger_decoded)
plt.show()



class Physical_simulation():
    def __init__(self, args):
        self.args = args
        self.M = args.M
        self.N = args.N

    def channel(self, data_x):
        inputBS, len_data_x = data_x.size(0), data_x.size(1)
        noise_std = 10 ** (-self.args.snr * 1.0 / 10 / 2)
        # real channel
        AWGN = torch.normal(0, std=noise_std, size=(inputBS, len_data_x), requires_grad=False).to(self.args.device)

        if self.args.fading == 1:
            # at the receiver, the equivalent noise power is reduced by self.hh**2
            self.hh = np.random.rayleigh(self.args.hstd, 1)  # sigma and number of samples
        else:
            self.hh = np.array([1])

        data_r = torch.from_numpy(self.hh).type(torch.float32).to(self.args.device) * data_x + AWGN
        return data_r
'''