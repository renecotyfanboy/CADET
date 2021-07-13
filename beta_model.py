from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from numpy import amin, arctan, argsort, array, cos, cumsum, deg2rad, dot, floor, hypot, indices, linspace, meshgrid, ones, pi, random, save, sqrt, sin, where, zeros
from numpy import sum as asum
from astropy.convolution import Gaussian2DKernel as Gauss
from astropy.convolution import convolve
from astropy.nddata import CCDData


class beta_model:
    def __init__(self, size, res, r0, ampl, beta, ampl2, r02, beta2,\
                 ellip, phi, axes, sloshing, point_source):

        self.res = res
        self.size = size
        self.center = (size - 0.5) * res * ones(3)
        self.points = int(2 * size * res)
        self.shape = (self.points, self.points, self.points)
        
        self.r0 = r0
        self.ampl = ampl
        self.beta = beta
        self.ellip = ellip
        self.ampl2 = ampl2
        self.r02 = r02
        self.beta2 = beta2
        self.phi = phi
        self.axes = axes
        self.axis = 0
        self.dx, self.dy = 0, 0
        
        self.slosh_depth = sloshing[0]
        self.slosh_period = sloshing[1]
        self.slosh_dir = sloshing[2]
        self.slosh_angle = sloshing[3]
        
        self.point_rad = point_source[0]
        self.point_ampl = point_source[1]
        
        self.double = bool(ampl2)
        self.rims = False
        self.cavities = False
        self.sloshing = bool(sloshing[0])
        self.point_source = bool(point_source[0])
    
    def dither(self, dx, dy):
        self.center[0] += dx
        self.center[1] += dy
    
    def create_rotation_matrix(self, ax, ay, az, inverse=False):
        if inverse: ax, ay, az = -ax, -ay, -az

        Rx = array([[1, 0, 0],
                    [0, cos(ax), -sin(ax)],
                    [0, sin(ax), cos(ax)]])

        Ry = array([[cos(ay), 0, sin(ay)],
                   [0, 1, 0],
                   [-sin(ay), 0, cos(ay)]])

        Rz = array([[cos(az), -sin(az), 0],
                   [sin(az), cos(az), 0],
                   [0, 0, 1]])
        
        return dot(Rz, dot(Ry, Rx))
    
    def create_ellipsoidal_grid(self, shape, center, radii, angle):
        radii = array(radii)
        R = self.create_rotation_matrix(*angle)
        xi = tuple(linspace(0, s-1, s) - floor(0.5 * s) for s in shape)

        xi = meshgrid(*xi, indexing='ij')
        points = array(xi).reshape(3, -1)[::-1]

        points = dot(R, points).T

        grid_center = array(center) - 0.5*array(shape[::-1])
        grid_center = dot(R, grid_center)

        points = points[:, ::-1]
        grid_center = grid_center[::-1]
        radii = radii[::-1]

        dR = (points - grid_center)**2
        dR = dR / radii**2
        nR = asum(dR, axis=1).reshape(shape)

        return sqrt(nR)

    def create_ellipsoid(self, shape, center, radii, angle):
        r = self.create_ellipsoidal_grid(shape, center, radii, angle)
        return where(r <= 1, 1, 0)
    
    def create_rims(self, shape, center, radii1, radii2, angle, typ, angle_to_center):
        ell1 = self.create_ellipsoid(shape, center, radii1, angle)
        ell2 = self.create_ellipsoid(shape, center, radii2, angle)
    
        if typ == 1: angle[2] += pi / 2
        elif typ == 1: 
            angle[2] = pi / 2 - angle_to_center
            radii1[1] *= 0.75
            radii2[1] *= 0.75
            
        r = self.create_ellipsoidal_grid(shape, center, radii2, angle)
        
        r1, r2 = min(radii1), min(radii2)
        alpha, beta = - 1 / (r1/r2 - 1), 1 / (r1/r2 - 1)
        model = alpha + beta * sqrt(r)

        shell = (ell2 - ell1) * model
        shell = where(shell < 0, 0, shell)
        shell = where(shell > 1, 1, shell)
        return shell
    
    def create_model(self):
        radii = array([1, 1-self.ellip, 1])
        angles = deg2rad([0, 0, self.phi])
        r = self.create_ellipsoidal_grid(self.shape, self.center, radii, angles)
        self.model3D = self.ampl * (1 + (r/self.r0)**2)**(-3/2*self.beta)
        if self.double:
            self.model3D += self.ampl2 * (1 + (r/self.r02)**2)**(-3/2*self.beta2)
        if self.point_source:
            self.model3D += self.point_ampl * (1 + (r/self.point_rad)**2)**(-3/2*self.beta*3)
        
    def cavity_pair(self, distances, phi, theta, radii, ellips, varphi, rims):
        self.res = 2
        d1 = distances[0] / self.res
        d2 = distances[1] / self.res
        
        phi1 = deg2rad(phi[0])
        phi2 = deg2rad(phi[1])
        theta1 = deg2rad(theta[0])
        theta2 = deg2rad(theta[1])
        varphi1 = deg2rad(varphi[0])
        varphi2 = deg2rad(varphi[1])
        
        x1 = self.center[0] - self.dx + d1 * cos(phi1) * cos(theta1)
        y1 = self.center[1] - self.dy + d1 * sin(phi1) * cos(theta1)
        z1 = self.center[2] + d1 * sin(theta1)
        x2 = self.center[0] - self.dx + d2 * cos(phi2) * cos(theta2)
        y2 = self.center[1] - self.dy + d2 * sin(phi2) * cos(theta2)
        z2 = self.center[2] + d2 * sin(theta2)
                
        rx1 = radii[0] / self.res
        ry1 = radii[0] * (1 - ellips[0]) / self.res
        rz1 = max(rx1, ry1)

        rx2 = radii[1] / self.res
        ry2 = radii[1] * (1 - ellips[1]) / self.res
        rz2 = max(rx2, ry2)
                        
        self.cav3D = self.create_ellipsoid(self.shape, [x1, y1, z1], [rx1, ry1, rz1], varphi1)
        self.cav3D += self.create_ellipsoid(self.shape, [x2, y2, z2], [rx2, ry2, rz2], varphi2)
        self.cavities = True

        if rims[0]:
            f = 1 + rims[0]
            rim1 = self.create_rims(self.shape, [x1, y1, z1], [rx1, ry1, rz1], [f*rx1, f*ry1, f*rz1], varphi1, rims[2], phi1)
            rim2 = self.create_rims(self.shape, [x2, y2, z2], [rx2, ry2, rz2], [f*rx2, f*ry2, f*rz2], varphi2, rims[2], phi2)
            self.rim3D = ones(self.shape) + rims[1] * rim1 + rims[1] * rim2
            self.rims = True
            
    def apply_mask(self):
        self.create_model()
        if self.cavities: 
            self.masked3D = where(self.cav3D > 0, 0, self.model3D)
            if self.rims: self.masked3D = self.masked3D * self.rim3D
            else: self.rim3D = zeros(self.shape)
        else: 
            self.masked3D = self.model3D
            self.rim3D = zeros(self.shape)
            self.cav3D = zeros(self.shape)
            
    def apply_sloshing(self):
        R = linspace(-1, 1, self.points)
        x, y = meshgrid(R, R)
        r = sqrt(x**2 + y**2) * pi * self.slosh_period
        if self.slosh_dir: x, y = y, x
        if self.slosh_angle == 90: self.slosh_angle += 1e-5
        rotation = deg2rad(self.slosh_angle)
        x, y = x * cos(rotation) - y * sin(rotation), x * sin(rotation) + y * cos(rotation)
        phi = arctan(y / x)
        angle = where(x > 0, phi + r, phi + r + pi)
        val = cos(angle) * self.slosh_depth + ones(x.shape)
        self.slosh = where(val < 0, 0, val)
        self.image *= self.slosh
        
    def apply_noise(self):
        self.apply_mask()
        self.image = asum(self.masked3D, axis=self.axis)

        if self.sloshing: self.apply_sloshing()
        else: self.slosh = zeros(self.image.shape)
            
        self.rim = asum(self.rim3D, axis=self.axis)
        self.mask = asum(self.cav3D, axis=self.axis)
        
        self.noisy = random.poisson(self.image)
    
    def plot_mask(self):
        plt.imshow(self.mask)
        plt.axis('off')
        
    def plot_rim(self):
        plt.imshow(self.rim)
        plt.axis('off')
        
    def plot_slosh(self):
        plt.imshow(self.slosh)
        plt.axis("off")
        
    def plot_binary_mask(self):
        plt.imshow(where(self.mask > 0, 1, 0))
        plt.axis('off')
        
    def plot_masked_image(self):
        plt.imshow(self.image, norm=LogNorm())
        plt.axis('off')
        
    def plot_masked_noisy_image(self):
        min_val = amin(where(self.noisy == 0, 10, self.noisy))
        img = where(self.noisy < min_val, min_val, self.noisy)
        plt.imshow(img, norm=LogNorm())
        plt.axis('off')
    
    def plot_smoothed_masked_image(self):
        kernel = Gauss(x_stddev = 1, y_stddev = 1, x_size = 11, y_size = 11)
        smoothed_noisy_img = convolve(self.noisy, boundary = "wrap", kernel = kernel)
       
        min_val = amin(where(smoothed_noisy_img == 0, 10, smoothed_noisy_img))
        img = where(smoothed_noisy_img < min_val, min_val, smoothed_noisy_img)
        plt.imshow(img, norm=LogNorm())
        plt.axis('off')
    
    def save_image(self, direct, i, fits=False):
        if fits:
            ccd = CCDData(self.noisy, unit="adu")
            ccd.write("{0}/{1}.fits".format(direct, i), overwrite=True)
        else:
            save("{0}/{1}_img".format(direct, i), self.noisy)
    
    def save_mask(self, direct, i, fits=False):
        if fits:
            ccd = CCDData(self.mask, unit="adu")
            ccd.write("{0}/{1}.fits".format(direct, i), overwrite=True)
        else:
            save("{0}/{1}_mask".format(direct, i), self.mask)

    def save_binary_mask(self, direct, i, fits=False):
        if fits:
            ccd = CCDData(where(self.mask > 0, 1, 0), unit="adu")
            ccd.write("{0}/{1}.fits".format(direct, i), overwrite=True)
        else:
            save("{0}/{1}_mask".format(direct, i), where(self.mask > 0, 1, 0))
    
    def plot_profile(self):
        plt.plot(self.azim_averaged(self.image))
        plt.xscale("log"); plt.yscale("log")
        plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
        plt.tick_params(axis='y',which='both',labelleft=False, left=False)
        
    def azim_averaged(self, image, center=None):
        y, x = indices(image.shape)

        if not center:
            center = array([(x.max()-x.min())/2.0-self.dx, (x.max()-x.min())/2.0-self.dy])

        r = hypot(x - center[0], y - center[1])
        ind = argsort(r.flat)
        r_sorted = r.flat[ind]
        i_sorted = image.flat[ind]
        r_int = r_sorted.astype(int)

        deltar = r_int[1:] - r_int[:-1]
        rind = where(deltar)[0]
        nr = rind[1:] - rind[:-1]

        csim = cumsum(i_sorted, dtype=float)
        tbin = csim[rind[1:]] - csim[rind[:-1]]
        radial_prof = tbin / nr

        return radial_prof