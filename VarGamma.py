# Variance Gamma distribution for python
#
# =======================================================================================
#
# Parameters of VarGamma distribution
# c - location
# sigma - spread
# theta - asymmetry
# nu - shape
# (according to Seneta, E. (2004) Fitting the variance-gamma model to financial data)
#
# =======================================================================================
#
# Required packages: numpy, scipy
#
# =======================================================================================
#
# Functions of the module:
#
# pdf(x=0.0, c=0.0, sigma=1.0, theta=0.0, nu=1.0)
#     evaluates VarGamma probability density function
#
# cdf(x=0.0, c=0.0, sigma=1.0, theta=0.0, nu=1.0)
#     evaluates VarGamma cumulative density function
#
# rnd(n=1, c=0.0, sigma=1.0, theta=0.0, nu=1.0)
#     generates n random points from VarGamma distribution
#
# fit(data)
#     synonym for fit_ml(data)
#
# fit_moments(x)
#     fits the parameters of VarGamma distribution to a given list of points
#     via method of moments, assumes that theta is small (so theta^2 = 0)
#
# fit_ml(data)
#     fits the parameters of VarGamma distribution to a given list of points
#     via Maximizing the Likelihood functional
#
# =======================================================================================

from numpy import *
from scipy import special
from scipy.integrate import quad
from scipy import optimize

def pdf_one_point(x=0.0, c=0.0, sigma=1.0, theta=0.0, nu=1.0):
	''' VarGamma probability density function in a point x '''
	temp1 = 2.0 / ( sigma*(2.0*pi)**0.5*nu**(1/nu)*special.gamma(1/nu) )
	temp2 = ((2*sigma**2/nu+theta**2)**0.5)**(0.5-1/nu)
	temp3 = exp(theta*(x-c)/sigma**2) * abs(x-c)**(1/nu - 0.5)
	temp4 = special.kv(1/nu - 0.5, abs(x-c)*(2*sigma**2/nu+theta**2)**0.5/sigma**2)
	return temp1*temp2*temp3*temp4

def pdf(x=0.0, c=0.0, sigma=1.0, theta=0.0, nu=1.0):
	''' VarGamma probability density function of an array or a point x '''
	if isinstance(x, (int, float, double)): # works with lists and arrays
		return pdf_one_point(x, c, sigma, theta, nu)
	else:
		return map(lambda x: pdf_one_point(x, c, sigma, theta, nu), x)

def cdf_one_point(x=0.0, c=0.0, sigma=1.0, theta=0.0, nu=1.0):
	''' VarGamma cumulative distribution function in a point x '''
	return quad(lambda x: pdf_one_point(x, c, sigma, theta, nu), -500, x, epsabs=1e-3)[0] # todo: analytical solution?

def cdf(x=0.0, c=0.0, sigma=1.0, theta=0.0, nu=1.0):
	''' VarGamma cumulative distribution function of an array or a point x '''
	if isinstance(x, (int, float,double)):
		return cdf_one_point(x, c, sigma, theta, nu)
	else:
		return map(lambda x: cdf_one_point(x, c, sigma, theta, nu), x)

def rnd(n=1, c=0.0, sigma=1.0, theta=0.0, nu=1.0):
	''' generates n random points from VarGamma distribution '''
	# build grid
	eps = 10e-7
	range_left = range_right = 1
	while pdf(c-range_left, c, sigma, theta, nu) > eps:
		range_left *= 2
	while pdf(c+range_right, c, sigma, theta, nu) > eps:
		range_right *= 2
	step = (range_right + range_left) / 1000.0 # todo: change to adaptive?
	grid = arange(c-range_left, c+range_right, step)
	pdf_values = pdf(grid, c, sigma, theta, nu)
	cdf_values = pdf_values
	for i in range(1, len(cdf_values)):
		cdf_values[i] += cdf_values[i-1]
	cdf_values /= max(cdf_values)
	# select the number which probability is close to r01
	r01 = random.rand(n) # random values from [0, 1]
	r = random.rand(n)
	for k in range(n):
		i = 1
		while cdf_values[i] < r01[k]:
			i += 1
		r[k] = grid[i-1] + (grid[i] - grid[i-1]) * random.rand()
	if n == 1:
		return r[0]
	else:
		return list(r)
	
def fit_moments(x):
	''' fits the parameters of VarGamma distribution to a given list of points
		via method of moments, assumes that theta is small (so theta^2 = 0)
		see: Seneta, E. (2004). Fitting the variance-gamma model to financial data. '''
	mu = mean(x)
	sigma_squared = mean( (x-mu)**2 )
	beta = mean( (x-mu)**3 ) / mean( (x-mu)**2 )**1.5
	kapa = mean( (x-mu)**4 ) / mean( (x-mu)**2 )**2
	# solve combined equations
	sigma = sigma_squared**0.5
	nu = kapa/3.0 - 1.0
	theta = sigma*beta / (3.0*nu)
	c = mu - theta
	return (c, sigma, theta, nu)

def neg_log_likelihood(data, par):
	''' negative log likelihood function for VarGamma distribution '''
	# par = array([c, sigma, theta, nu])
	if (par[1] > 0) & (par[3] > 0):
		return -sum(log( pdf(data, c=par[0], sigma=par[1], theta=par[2], nu = par[3]) ))
	else:
		return Inf

def fit_ml(data):
	''' fits the parameters of VarGamma distribution to a given list of points
		via Maximizing the Likelihood functional (=minimizing negative log likelihood)
		the initial point is chosen with fit_moments(x),
		optimization is performed using Nedler-Mead method '''
	par_init = array( fit_moments(data) )
	par = optimize.fmin(lambda x: neg_log_likelihood(data, x), par_init, maxiter=100)
	return tuple(par)

def fit(data):
	''' is equivalent to fit_ml '''
	return fit_ml(data)
