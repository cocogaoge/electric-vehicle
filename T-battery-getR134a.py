from CoolProp.CoolProp import PropsSI

H = PropsSI('H', 'T', 298.15, 'P', 101325, 'R134a')
print("H =", H)#J/kg
Tc = PropsSI('T', 'P', 106400, 'Q', 1, 'R134a')
print("Tc =", Tc-273.15)#K
