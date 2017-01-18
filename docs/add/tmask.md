**Tmask Function**  

Uses Robust Regression method to fit curves for the green SWIR1 and bands,
for a given observation window.

Inputs: dates, green and SWIR1 bands observations, median variogram values for
green and SWIR1 bands, T_CONST

set up coefficient matrix

[RobustFit] (robust_fit.md) green band
[RobustFit] (robust_fit.md) SWIR1 band

set up tmask accummulator for outliers

delta green band = observed values – predicted values
delta SWIR1 band = observed values – predicted values

Tmask value = 1
&nbsp;&nbsp;&nbsp;&nbsp;where ( delta green band > median variogram green band * T_Const )
&nbsp;&nbsp;&nbsp;&nbsp;or    ( delta SWIR1 band > median variogram SWIR1 band * T_Const )

else Tmask value = 0

return outliers
