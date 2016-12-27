**Method 1 (standard operation)**

1. Mask out repeated date values, and their associated spectral values  
 - Matlab code does not make any preference on which one to keep  
2. Calculate a modified first-order variogram/madogram across each spectral band  
 - Median of the absolute value of the discrete differences of spectral values  
 ```python  
 >>> import numpy as np  
 >>> vario[band_2_idx] = np.median(numpy.abs(numpy.diff(band_2)))  
 ```  
3. Mask out values outside of acceptable ranges  

while there are clear observations:  

&nbsp;&nbsp;&nbsp;&nbsp;if there are enough observations and minimum temporal span:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;calculate [Tmask] (tmask.md) and mask out where Tmask is 1  

&nbsp;&nbsp;&nbsp;&nbsp;if not enough observations or minimum temporal span after Tmask:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;continue to the next observation  

&nbsp;&nbsp;&nbsp;&nbsp;initial [LASSOFit(observations, df=4)] (lasso_fit.md)  

&nbsp;&nbsp;&nbsp;&nbsp;#check for a stable start  
&nbsp;&nbsp;&nbsp;&nbsp;If [change detected] (detect_change.md):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model start position += 1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;not a stable start, continue to next observation  

&nbsp;&nbsp;&nbsp;&nbsp;# Stable start  
&nbsp;&nbsp;&nbsp;&nbsp;# Move backward to see if previous values fall within the new curve  

