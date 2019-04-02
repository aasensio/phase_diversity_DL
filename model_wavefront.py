import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import wavefront as wf
from ipdb import set_trace as stop

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, upsample=False):
        super(ConvBlock, self).__init__()

        self.upsample = upsample

        if (upsample):
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=1)
        else:
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride)

        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)

        self.reflection = nn.ReflectionPad2d(int((kernel_size-1)/2))
        self.bn = nn.BatchNorm2d(inplanes)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.bn(x)
        out = self.activation(out)
        
        if (self.upsample):
            out = torch.nn.functional.interpolate(out, scale_factor=2)

        out = self.reflection(out)
        out = self.conv(out)        
            
        return out

class encoder(nn.Module):
    def __init__(self, in_planes, out_planes, npix_out, n=32):
        super(encoder, self).__init__()

        self.npix_out = npix_out

        self.A01 = ConvBlock(in_planes, n, kernel_size=9)
        
        self.C01 = ConvBlock(n, 2*n, stride=2)
        self.C02 = ConvBlock(2*n, 2*n)
        self.C03 = ConvBlock(2*n, 2*n)        
        self.C04 = ConvBlock(2*n, 2*n)
        self.drop_C04 = nn.Dropout(0.3, inplace=True)

        self.C11 = ConvBlock(2*n, 4*n, stride=2)
        self.C12 = ConvBlock(4*n, 4*n)
        self.C13 = ConvBlock(4*n, 4*n)        
        self.C14 = ConvBlock(4*n, 4*n)
        self.drop_C14 = nn.Dropout(0.3, inplace=True)
        
        self.C21 = ConvBlock(4*n, 8*n, stride=2)
        self.C22 = ConvBlock(8*n, 8*n)
        self.C23 = ConvBlock(8*n, 8*n)        
        self.C24 = ConvBlock(8*n, 8*n)
        self.drop_C24 = nn.Dropout(0.3, inplace=True)
        
        self.C31 = ConvBlock(8*n, 4*n, upsample=True)
        self.C32 = ConvBlock(4*n, 4*n)
        self.C33 = ConvBlock(4*n, 4*n)
        self.C34 = ConvBlock(4*n, 4*n)
        
        self.C41 = ConvBlock(4*n, 2*n, upsample=True)
        self.C42 = ConvBlock(2*n, 2*n)
        self.C43 = ConvBlock(2*n, 2*n)
        self.C44 = ConvBlock(2*n, 2*n)
        
        self.C51 = ConvBlock(2*n, n, upsample=True)
        self.C52 = ConvBlock(n, n)
        self.C53 = ConvBlock(n, n)
        self.C54 = ConvBlock(n, n)

        self.C61 = ConvBlock(n, n)
        self.C62 = ConvBlock(n, n)

        self.C63 = nn.Conv2d(n, out_planes, kernel_size=1, stride=1)
        nn.init.kaiming_normal_(self.C63.weight)
        nn.init.constant_(self.C63.bias, 0.1)                    
        
    def forward(self, x):
        A01 = self.A01(x)

        # N -> N/2
        C01 = self.C01(A01)
        C02 = self.C02(C01)
        C03 = self.C03(C02)        
        C04 = self.C04(C03)
        #C04 = self.drop_C04(C04)
        C04 += C01
        
        # N/2 -> N/4
        C11 = self.C11(C04)
        C12 = self.C12(C11)
        C13 = self.C13(C12)        
        C14 = self.C14(C13)
        #C14 = self.drop_C14(C14)
        C14 += C11
        
        # N/4 -> N/8
        C21 = self.C21(C14)
        C22 = self.C22(C21)
        C23 = self.C23(C22)        
        C24 = self.C24(C23)
        #C24 = self.drop_C24(C24)
        C24 += C21
        
        # N/8 -> N/4
        C31 = self.C31(C24)
        # C31 += C14
        C32 = self.C32(C31)
        C33 = self.C33(C32)        
        C34 = self.C34(C33)
        C34 += C31

        # N/4 -> N/2
        C41 = self.C41(C34)
        # C41 += C04
        C42 = self.C42(C41)
        C43 = self.C43(C42)        
        C44 = self.C44(C43)
        C44 += C41

        # N/2 -> N
        C51 = self.C51(C44)
        C52 = self.C52(C51)
        C53 = self.C53(C52)        
        C54 = self.C54(C53)
        C54 += C51

        # N -> 2N
        #C61 = self.C61(C54)
        #C62 = self.C62(C61)
        #out = self.C63(C62)

        out = nn.functional.interpolate(C54, size=(self.npix_out, self.npix_out), mode='bilinear', align_corners=False)
        out = self.C61(out)
        out = self.C62(out)
        out = self.C63(out)

        return out

class network(nn.Module):
    def __init__(self, telescope_diameter, pixel_size, lambda0, npix_psf, device, batchsize, n_zernike=0, architecture='encdec_64'):
        super(network, self).__init__()

        # Define all parameters for correctly carrying out the convolution with generated PSFs
        self.lambda0 = lambda0
        self.telescope_diameter = telescope_diameter
        self.pixel_size = pixel_size		
        self.npix_psf = npix_psf
        self.n_zernike = n_zernike
        self.batchsize = batchsize
        self.device = device

        # Compute the PSF scale appropriate for the required pixel size, wavelength and telescope diameter
        self.overfill = wf.psfScale(self.telescope_diameter, self.lambda0, self.pixel_size)

        self.nbig = int(np.ceil(self.npix_psf * self.overfill))
        if (self.nbig % 2 != 0):
            self.nbig += 1

        self.half = (self.nbig - self.npix_psf) // 2
        center = self.nbig // 2

        if (self.n_zernike != 0):
            self.zernikes = np.zeros((self.n_zernike,self.npix_psf,self.npix_psf))
            for j in range(self.n_zernike):
                self.zernikes[j,:,:] = wf.zernike(j+4,npix=self.npix_psf) 

        self.zernikes_torch = torch.zeros((self.n_zernike,self.nbig,self.nbig),dtype=torch.float32).to(self.device)

        self.zernikes_torch[:,self.half:self.half+self.npix_psf,self.half:self.half+self.npix_psf] = torch.from_numpy(self.zernikes)
        
        
        # Temporary arrays for PSF calculation and convolution
        self.wfbig = torch.zeros((batchsize,self.nbig,self.nbig),dtype=torch.float32).to(self.device)
        self.illum = torch.zeros((batchsize,self.nbig,self.nbig),dtype=torch.float32).to(self.device)
        self.illum_grad_wavefront = torch.zeros((batchsize,2,self.nbig,self.nbig),dtype=torch.float32).to(self.device)
        self.phase = torch.zeros((batchsize,self.nbig,self.nbig,2),dtype=torch.float32).to(self.device)
        self.psf = torch.zeros((batchsize,self.npix_psf,self.npix_psf,2),dtype=torch.float32).to(self.device)
        self.image = torch.zeros((batchsize,self.npix_psf,self.npix_psf,2),dtype=torch.float32).to(self.device)
        self.prod = torch.zeros((batchsize,self.npix_psf,self.npix_psf,2),dtype=torch.float32).to(self.device)
        self.zernike_defocus = torch.zeros((batchsize,self.nbig,self.nbig),dtype=torch.float32).to(self.device)

        self.zeros = torch.zeros((self.batchsize, self.npix_psf, self.npix_psf, 1),dtype=torch.float32).to(self.device)

        # Defocus Zernike coefficient. It is at index 0 because we are not considering modes 1,2 and 3 (piston, tip, tilt)
        self.zernike_defocus[:,:,:] = 1.0 * np.pi / np.sqrt(3.0) * self.zernikes_torch[0,:,:][None,:,:]

        # Compute telescope aperture
        self.aperture = wf.aperture(npix = self.npix_psf, cent_obs = 0, spider=0)

        # Illumination of the pupil
        self.illum[:,self.half:self.half+self.npix_psf,self.half:self.half+self.npix_psf] = torch.from_numpy(self.aperture)[None,:,:]
        self.illum_grad_wavefront[:,:,self.half:self.half+self.npix_psf,self.half:self.half+self.npix_psf] = torch.from_numpy(self.aperture)[None,None,:,:]

        # Compute mask to select the central part of the PSF, which has been computed in a larger
        # array to fit the pixel size with the spatial frequencies
        self.mask = torch.ones((self.batchsize, self.nbig, self.nbig)).byte().to(self.device)
        

        self.mask[:,center-self.half:center+self.half+1,:] = 0
        self.mask[:,:,center-self.half:center+self.half+1] = 0

        # Define the encoder network
        print('Architecture : {0}'.format(architecture))
        if (architecture == 'encdec_64'):
            self.deepnet = encoder(in_planes=2, out_planes=1, npix_out=self.npix_psf, n=32)
            self.cut_left = 18
            self.cut_right = 82

        if (architecture == 'encdec_128'):
            self.deepnet = encoder(in_planes=2, out_planes=1, npix_out=self.npix_psf, n=32)
            self.cut_left = 21
            self.cut_right = -22

        if (architecture == 'keepsize'):
            self.deepnet = keepsize(in_planes=2, out_planes=1)

        # Sobel filters
        gx = torch.Tensor([[1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]]).view(1,1,3,3)

        gy = torch.Tensor([[1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]]).view(1,1,3,3)

        self.sobel = torch.cat([gx,gy], dim=0).to(self.device)
                        
    def forward(self, focus_defocused, original_large, zernike_target):
        
        self.image[:,:,:,0] = original_large
        image_ft = torch.fft(self.image, 2)

        wavefront_target = self.illum * torch.sum(zernike_target[:,:,None,None] * self.zernikes_torch, 1)

        # Compute wavefront with neural network
        wavefront = self.deepnet(focus_defocused)

        # Before padding the wavefront to the final size, compute gradients using Sobel filters for regularization
        grad_wavefront = nn.functional.conv2d(wavefront, self.sobel, padding=1)

        # Multiply by the aperture and select small size
        grad_wavefront = self.illum_grad_wavefront * nn.functional.pad(grad_wavefront, (self.half, self.nbig-(self.half+self.npix_psf),self.half, self.nbig-(self.half+self.npix_psf)))
        grad_wavefront = grad_wavefront[:,:,self.half:self.half+self.npix_psf,self.half:self.half+self.npix_psf]

        # Now pad the wavefront ot the needed size
        wavefront = nn.functional.pad(wavefront, (self.half, self.nbig-(self.half+self.npix_psf),self.half, self.nbig-(self.half+self.npix_psf)))

        # Remove singleton dimensions and multiply by the aperture
        wavefront = self.illum * torch.squeeze(wavefront)

        # Compute the Zernike coefficients from the focused+defocused pair
        #zernike = torch.squeeze(self.encoder(focus_defocused))

        #----------------
        # Focused image
        #----------------
        # Compute wavefront by summing over Zernike polynomials
        #wavefront = torch.sum(zernike[:,:,None,None] * self.zernikes_torch, 1)

        # Compute real and imaginary parts of the pupil
        tmp1 = torch.unsqueeze(torch.cos(wavefront) * self.illum, -1)
        tmp2 = torch.unsqueeze(torch.sin(wavefront) * self.illum, -1)

        self.phase = torch.cat([tmp1, tmp2], -1)

        # Compute FFT of the pupil function and compute autocorrelation
        ft = torch.fft(self.phase, 2)
        psf = ft[:,:,:,0]**2 + ft[:,:,:,1]**2

        # Extract the central part of the PSF of the size of the input undegraded image
        psf = psf[self.mask].view((self.batchsize, self.npix_psf, self.npix_psf))
        
        # Normalize PSF and transform to pytorch-complex
        tmp = torch.unsqueeze(psf / torch.sum(psf, [1,2])[:,None,None], -1)

        self.psf = torch.cat([tmp, self.zeros], -1)
        
        # Compute convolution with PSF
        psf_ft = torch.fft(self.psf, 2)

        tmp1 = torch.unsqueeze(psf_ft[:,:,:,0] * image_ft[:,:,:,0] - psf_ft[:,:,:,1] * image_ft[:,:,:,1], -1)
        tmp2 = torch.unsqueeze(psf_ft[:,:,:,0] * image_ft[:,:,:,1] + psf_ft[:,:,:,1] * image_ft[:,:,:,0], -1)

        self.prod = torch.cat([tmp1, tmp2], -1)
        
        out_focused = torch.ifft(self.prod, 2)[:,self.cut_left:self.cut_right,self.cut_left:self.cut_right,0]

        #----------------
        # Defocused image
        #----------------
        wavefront_defocus = (wavefront + self.zernike_defocus)

        # Compute real and imaginary parts of the pupil
        tmp1 = torch.unsqueeze(torch.cos(wavefront_defocus) * self.illum, -1)
        tmp2 = torch.unsqueeze(torch.sin(wavefront_defocus) * self.illum, -1)

        self.phase = torch.cat([tmp1, tmp2], -1)

        # Compute FFT of the pupil function and compute autocorrelation
        ft = torch.fft(self.phase, 2)
        psf = ft[:,:,:,0]**2 + ft[:,:,:,1]**2

        # Extract the central part of the PSF of the size of the image
        psf = psf[self.mask].view((self.batchsize, self.npix_psf, self.npix_psf))
        
        # Normalize PSF and transform to pytorch-complex
        tmp = torch.unsqueeze(psf / torch.sum(psf, [1,2])[:,None,None], -1)

        self.psf = torch.cat([tmp, self.zeros], -1)
        
        # Compute convolution with PSF
        psf_ft = torch.fft(self.psf, 2)

        tmp1 = torch.unsqueeze(psf_ft[:,:,:,0] * image_ft[:,:,:,0] - psf_ft[:,:,:,1] * image_ft[:,:,:,1], -1)
        tmp2 = torch.unsqueeze(psf_ft[:,:,:,0] * image_ft[:,:,:,1] + psf_ft[:,:,:,1] * image_ft[:,:,:,0], -1)

        self.prod = torch.cat([tmp1, tmp2], -1)
        
        out_defocused = torch.ifft(self.prod, 2)[:,self.cut_left:self.cut_right,self.cut_left:self.cut_right,0]
        
        return out_focused, out_defocused, wavefront, wavefront_target, grad_wavefront