import VarGamma as vg
from numpy import *
import matplotlib.pyplot as plt

random.seed(1)

c = 0.0     # location
sigma = 1.0 # spread
theta = 0.4 # asymmetry
nu = 0.8    # shape

grid = arange(-10, 10, 0.1)
pdf_values = vg.pdf(grid, c, sigma, theta, nu)
cdf_values = vg.cdf(grid, c, sigma, theta, nu)
data = vg.rnd(100, c, sigma, theta, nu)

# try fitting parameters
print 'true parameters:'
print (c, sigma, theta, nu)
print 'parameters estimated by Methods of Moments:'
(c_fit, sigma_fit, theta_fit, nu_fit) = vg.fit_moments(data)
print (c_fit, sigma_fit, theta_fit, nu_fit)
print 'parameters estimated by Maximum Likelihood:'
(c_fit, sigma_fit, theta_fit, nu_fit) = vg.fit(data)
print (c_fit, sigma_fit, theta_fit, nu_fit)

# prepare plotting tools
fig = plt.figure()
plt.hold(True)
plt.axis('off')
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# plot histogram of random data
hist,bins = histogram(data, bins=20)
hist = double(hist)
hist *= max(pdf_values) / max(hist) # just normalisation
width = 0.5*(bins[1] - bins[0])
center = (bins[:-1]+bins[1:]) / 2
ax1.bar(center, hist, align='center', width=width)

# plot pdf of the distribution
ax1.plot(grid, pdf_values, linewidth=3, color='r')

# plot cdf of the distribution
ax2.plot(grid, cdf_values, linewidth=3, color='g')

ax1.set_title('VarGamma PDF and the histogram of random points')
ax2.set_title('VarGamma CDF')
plt.show()
