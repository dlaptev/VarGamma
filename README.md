VarGamma
========

Variance Gamma (VarGamma) distribution module for Python.

**Implements:**
  * probability density function,
  * cumulative distribution function,
  * random point generator,
  * two parameter fitting methods (method of moments and maximum likelihood).

**Parameters** of VarGamma distribution according to Seneta, E. "Fitting the variance-gamma model to financial data" (2004):
  * c - location
  * sigma - spread
  * theta - asymmetry
  * nu - shape

**Requires:**
  * numpy,
  * scipy

See *example.py* for more details.
