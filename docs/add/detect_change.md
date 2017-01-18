**Detect Change in Select Bands**  
Inputs: per band observations, current models, start date, end date, median band variogram values, T_Cg value

for each band:
&nbsp;&nbsp;&nbsp;&nbsp;minimum rmse = max ( median band variogram, model rmse )

&nbsp;&nbsp;&nbsp;&nbsp;slope = model slope coefficient * ( end date - start date )

&nbsp;&nbsp;&nbsp;&nbsp;band check val = ( abs ( slope ) + abs ( first residual ) + abs ( last residual ) ) / minimum rmse

if the norm ( band check vals ) **2 > T_cg:
&nbsp;&nbsp;&nbsp;&nbsp;change detected
