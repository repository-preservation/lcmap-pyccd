**LASSO Regression ( Least Absolute Shrinkage and Selector Operator )**  

Lasso regression is mathematical model to do curve fitting of points.
The python module is a standard all python apps/modules adhere to.
Calculations are documented and consistent across platforms and applications.

Inputs: observations, degrees of freedom ( mimimum number of coefficients )

W = 2 * pi / 365.25

Independent_arr [ column1 ] = observations

if number of coefficients >= 4:
&nbsp;&nbsp;&nbsp;&nbsp;Independent_arr [ column2 ] = cos ( W * d ) for d in dates
&nbsp;&nbsp;&nbsp;&nbsp;Independent_arr [ column3 ] = sin ( W * d ) for d in dates

If df >= 6:
&nbsp;&nbsp;&nbsp;&nbsp;Independent_arr [ column4 ] = cos ( 2 * W * d ) for d in dates
&nbsp;&nbsp;&nbsp;&nbsp;Independent_arr [ column5 ] = sin ( 2 * W * d ) for d in dates

If df >= 8:
&nbsp;&nbsp;&nbsp;&nbsp;Independent_arr [ column6 ] = cos ( 3 * W * d ) for d in dates
&nbsp;&nbsp;&nbsp;&nbsp;Independent_arr [ column7 ] = sin ( 3 * W * d ) for d in dates

Regress

Minimum number of observations required for a fit is defined as:
number of coefficients (4,6, or 8) * NUM_OBS_FACTOR (3)

Return fitted coefficients, RMSE, residuals ( actual - predicted )
