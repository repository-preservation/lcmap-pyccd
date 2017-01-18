Default Parameters:
(Parameters is not entirely accurate.  In practice, these values are in a
.yaml file, which is read, and can be adjusted, but these are not actual
“command line arguments”.)

- change probability threshold
T_cg = inverse of chi^2(0.99, number of detection bands)
```python
>>> from scipy.stats import chi2
>>> chi2.ppf(0.99, 5)
15.086272469388987
 ```
 
- outlier threshold (Tmasking of noise)
Tmax_cg = inverse of chi^2(1-1e-6, number of detection bands)
```python
>>> from scipy.stats import chi2
>>> chi2.ppf(1-1e-6, 5)
35.888186879610423
 ```

- Minimum Number of Consecutive Observations requied to identify a change
  6

- Maximum Number of Model Coefficients Produced
  8

- Clear count threhold percent
  0.25

- Snow count threhold percent
  0.75

- LASSO Regression lambda value
  20 (limiting cross-validation saves processing time)

Initial change detection setup:
 
- pyccd Input Landsat Data
  red, green, NIR, SWIR1, SWIR2, Thermal, cfmask

- Landsat bands for detection change
  red, green, NIR, SWIR1, SWIR2

- Mask out duplicate observations using the first of two swath overlap
  observations

- Sort observations temporally

- Create mask(s) of values that fall outside of acceptable ranges, considered
  saturated if > acceptable maximum:
  - Surface Reflectance bands acceptable range:           (  0, 10000 )
  - Brightness Temperature (Thermal) acceptable range: ( -9320,  7070 )

- Convert thermal values from kelvin to Celsius
    `val * 10 – 27315`

- Create variogram for each band

- Derive some QA (CFmask) masks/information for reference during the process
  - Clear pixel mask
    `val < 2 (water and clear)`
  - Non-Fill mask
    `val < 255`
  - Percent clear
    `sum(clear) / sum(non-fill)`
  - Snow mask
    `val == 3`
  - Percent snow
    `sum(snow) / (sum(clear) + sum(snow) + 0.01)`

if the percentage of clear pixels is less than the clear threshold:  
&nbsp;&nbsp;&nbsp;&nbsp;if the percentage of snow is greater than the snow threshold:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Persistent Snow Procedure] (snow.md)  
&nbsp;&nbsp;&nbsp;&nbsp;else:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Insufficient Clear Procedure] (fmask_fail.md)  
else:  
&nbsp;&nbsp;&nbsp;&nbsp;[Standard Procedure] (standard.md)  
