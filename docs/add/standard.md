**Standard Procedure**  

observations can and will be requested intelligently, but to protect a user
obtaining input data from different conventions, ensure it is pyccd-ready:
- mask out repeated observations, keeping first of two swath overlaps
- sort temporally
- convert thermal values from Kelvin to Celsius
- mask out values outside of acceptable ranges
- calculate a modified first-order variogram/madogram across each spectral band
 - Median of the absolute value of the discrete differences of spectral values
 ```python
 >>> import numpy as np
 >>> vario [ band_2_idx ] = np.median ( numpy.abs ( numpy.diff ( band_2 ) ) )
 ```  

while there are clear observations:  

&nbsp;&nbsp;&nbsp;&nbsp;if there are enough observations and minimum temporal span:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;calculate [Tmask] (tmask.md) and mask out where [Tmask] (tmask_md) is 1  

&nbsp;&nbsp;&nbsp;&nbsp;if not enough observations or minimum temporal span after [Tmask] (tmask_md):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;continue to the next observation  

&nbsp;&nbsp;&nbsp;&nbsp;initial [do generalized curve fit for observation set ( observations, number coeffs = 4 )] (lasso_fit.md)  

&nbsp;&nbsp;&nbsp;&nbsp;#check for a stable start  
&nbsp;&nbsp;&nbsp;&nbsp;if [change detected] (detect_change.md):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model start position += 1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;not a stable start, continue to next observation  

&nbsp;&nbsp;&nbsp;&nbsp;# stable start  
&nbsp;&nbsp;&nbsp;&nbsp;# move backward in time to see if previous values fit current curve  

&nbsp;&nbsp;&nbsp;&nbsp;if change detected:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;adjust model window start  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[build new curve fit] (lasso_fit.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;look forward in time to see if additional values fit curve  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if no change deteced:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if no change detected:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if new curve required:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;update stop, break, model new curve [LASSOFit]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;proceed to next obersvation  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;record change model  

if number of remaining observations < MEOW size but >= Peek Size:  
&nbsp;&nbsp;&nbsp;&nbsp;[fit generalized model for remaining observations] (lasso_fit.md)  
send/save results  
exit  
