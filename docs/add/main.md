Default Parameters:  

- Bands for detection change  
B_detect = 2 thru 6  

- change probability threshold  
T_cg = inverse of chi^2(0.99, number of detection bands)  
```python  
>>> from scipy.stats import chi2  
>>> chi2.ppf(0.99, 5)  
15.086272469388987  
 ```  
 
- number of consecutive observations  
conse = 6  

- maximum number of coefficients used  
max_c = 8  

- Clear count threhold
clear = 0.25

- Snow count threhold
snow = 0.75

- Tmasking of noise  
Tmax_cg = inverse of chi^2(1-1e-6, number of detection bands)  

Initial change detection setup:  

1. Convert thermal values from kelvin to Celsius  
    `val * 10 – 27315`

2. Create mask(s) of values that fall outside of acceptable ranges, considered saturated if > max  
    - Reflectance acceptable range: (0, 10000)  
    - Thermal acceptable range: (-9320, 7070)  

3. Derive some QA (CFmask) masks/information for reference during the process  
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

\*Inputs are inferred to be in temporal order  
If the percentage of clear pixels is less than the clear threshold:  
&nbsp;&nbsp;&nbsp;&nbsp;If the percentage of snow is greater than the snow threshold:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Snow Procedure] (snow.md)  
&nbsp;&nbsp;&nbsp;&nbsp;Else:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Fmask Fail Procedure] (fmask_fail.md)  
Else:  
&nbsp;&nbsp;&nbsp;&nbsp;[Standard Procedure] (standard.md)  
