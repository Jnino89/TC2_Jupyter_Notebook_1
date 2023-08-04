#-------------------------------------------------------------------------------------------------------------------#

# Inicializamos e importamos módulos

# Módulos externos
import matplotlib as mpl
import matplotlib.pyplot as plt
import math as m
import numpy as np
import scipy.signal as sig

fig_sz_x = 10
fig_sz_y = 9
fig_dpi = 150 # dpi

fig_font_size = 12

mpl.rcParams['figure.figsize'] = (fig_sz_x, fig_sz_y)
mpl.rcParams['figure.dpi'] = fig_dpi
#plt.rcParams.update({'font.size':fig_font_size})

#-------------------------------------------------------------------------------------------------------------------#

# Importamos las funciones de PyTC2

from pytc2.sistemas_lineales import analyze_sys
from pytc2.sistemas_lineales import pretty_print_lti

#-------------------------------------------------------------------------------------------------------------------#

# Datos de plantilla

alfa_max = 1                        # dB
alfa_min = 20                       # dB

f_s_plantilla = 1.2e3               # Hz
f_p_plantilla = 4.6e3               # Hz

omega_p_plantilla = 2*m.pi*f_p_plantilla
norm_w = omega_p_plantilla
omega_p_plantilla /= norm_w
omega_p_pb_prototipo = 1/omega_p_plantilla

omega_s_plantilla = 2*m.pi*f_s_plantilla
omega_s_plantilla /= norm_w
omega_s_pb_prototipo = 1/omega_s_plantilla

print('#---------------------------------------------------------------------------------------------#')
print(' ')
print('-- Datos de plantilla - Filtro pasa altos objetivo --')
print(' ')

print('α_min = {}'.format(alfa_min))
print('α_max = {}'.format(alfa_max))
print('fs = {}'.format(f_s_plantilla))
print('fp = {}'.format(f_p_plantilla))

print(' ')

print('ωp_n = {:3.4f}'.format(omega_p_plantilla))
print('ωs_n = {:3.4f}'.format(omega_s_plantilla))

print(' ')
print('-- Dominio transformado - Filtro pasa bajos prototipo --')
print(' ')

print('ωp = {:3.4f}'.format(omega_p_pb_prototipo))
print('ωs = {:3.4f}'.format(omega_s_pb_prototipo))

#-------------------------------------------------------------------------------------------------------------------#

# Calculamos e^2

print(' ')
print('#---------------------------------------------------------------------------------------------#')
print(' ')
print('-- ε: Grado de libertad de la funcion --')
print(' ')

print('ε = sqrt(10^(α_max/10)-1)')
eps = np.sqrt(10**(alfa_max/10)-1)

print(' ')

print('ε = {:3.4f}'.format(eps))
print('ε^2 = {:3.4f}'.format(eps**2))

print(' ')
print('#---------------------------------------------------------------------------------------------#')

#-------------------------------------------------------------------------------------------------------------------#

# Calculamos N iterando alfa_min

print(' ')
print('-- N: Orden del filtro --')
print(' ')
print('α_min = 10*log(1 + ε^2*C_n^2) con C_n^2= cosh^2(N*cosh^(-1)(ω_s))')
print(' ')

for N in range(1,10):
    C_nn = (np.cosh(N*(np.arccosh(omega_s_pb_prototipo))))**2
    att = 10*np.log10(1+(eps**2)*C_nn)    
    print('Para N= {:d}: ¿att = {:3.4f} dB >= α_min= {:d}?'.format(N, att, alfa_min))
    if att >= alfa_min:
        break

print(' ')

# Elegimos N de forma tal que α_min >= 20 dB:
print('N= {:d}'.format(N))

print(' ')
print('#---------------------------------------------------------------------------------------------#')

#-------------------------------------------------------------------------------------------------------------------#

# De la resolución analítica, obtenemos |T(jw)|^2 |w=s/j = 1 / (1 + ε^2*C_n^2) = T(s)*T(-s) 

#Cn2 = 2w^2 - 1                 # Polinomio grado 2 de Chevyshev
Cn2 = [2, 0, -1]                # Coeficientes del polinomio grado 2 de Chevyshev
Cn4 = np.polymul(Cn2, Cn2)      # Elevamos al cuadrado el denominador
Cn4 = Cn4 * eps**2              # Multiplicamos por ε^2
Cn4[4]= Cn4[4] + 1              # Sumamos 1 al término independiente
Cn4[2]= -Cn4[2]                 # Cambiamos de signo al coeficiente del término cuadrático producto del reeemplazo de w^2= s^2/j^2
Cn4 = Cn4 / (4*eps**2)          # Hacemos mónico el polinomio

p_T_lp= np.roots(Cn4)

den_T_lp= np.poly([p_T_lp for p_T_lp in p_T_lp if p_T_lp.real < 0])
num_T_lp= [m.sqrt(1/(4*eps**2))]
#print(num_T_lp)
#print(den_T_lp)

num_T_hp, den_T_hp= sig.lp2hp(num_T_lp, den_T_lp)
#print(num_T_hp)
#print(den_T_hp)

print(' ')
print('-- z_i: Ceros obtenidos con roots(p) del polinomio numerador --')
print('-- obtenido de forma analítica --')
print(' ')

print('z1,2= {}'.format(np.roots(num_T_hp)))

print(' ')
print('-- p_i: Polos en el semiplano izquierdo obtenidos con roots(p) --')
print('-- del polinomio denominador obtenido de forma analítica --')
print(' ')

print('p1,2= {}'.format(np.roots(den_T_hp)))

print(' ')
print('#---------------------------------------------------------------------------------------------#')

#-------------------------------------------------------------------------------------------------------------------#

# Obtenemos singularidades con la función de aproximación de Chevyshev del 
# filtro pasa bajos prototipo: cheb1ap(N, rp)

z, p, k = sig.cheb1ap(N, alfa_max)

#-------------------------------------------------------------------------------------------------------------------#

# Transformamos las singularidades que nos devuelve cheb1ap(N, rp) para obtener 
# el numerador y el denominador de la T_LP(s) del filtro pasa bajos prototipo 
# con zpk2tf(z,p,k):

num_lp, den_lp = sig.zpk2tf(z,p,k)

#-------------------------------------------------------------------------------------------------------------------#

# Transformamos la T_LP(s) del filtro pasa bajos prototipo para obtener 
# el numerador y el denominador de la T_HP(s) con lp2hp(num, den):

num_hp, den_hp = sig.lp2hp(num_lp, den_lp)

print(' ')
print('-- z_i: Ceros obtenidos con cheb1ap(N, rp) coincidentes --')
print('con los obtenidos con roots(p) de forma analítica --')
print(' ')

print('z1,2= {}'.format(np.roots(num_hp)))

print(' ')
print('-- p_i: Polos en el semiplano izquierdo obtenidos con cheb1ap(N, rp) --')
print('-- coincidentes con los obtenidos con roots(p) de forma analítica --')
print(' ')

print('p1,2= {}'.format(np.roots(den_hp)))

print(' ')
print('#---------------------------------------------------------------------------------------------#')

#-------------------------------------------------------------------------------------------------------------------#

# Transformamos (factorizamos) la T(s) en secciones de segundo orden (SOS) con la función tf2sos_analog()
#this_sos = tf2sos_analog(num_hp, den_hp)

t_T_hp= sig.TransferFunction(num_T_hp, den_T_hp)             # numerador y denominador resolución analítica

t_hp= sig.TransferFunction(num_hp, den_hp)                   # numerador y denominador con cheb1ap()

#-------------------------------------------------------------------------------------------------------------------#

# Imprimimos la T(s) en formato T(s)= num/den con la funcion pretty_print_lti(num, den):

print(' ')
print('-- Función transferencia T(s) analítica: --')
print(' ')

pretty_print_lti(num_T_hp, den_T_hp)

print(' ')
print('-- Función transferencia T(s) con cheb1ap(N, rp): --')
print(' ')

pretty_print_lti(num_hp, den_hp)

print(' ')
print('#---------------------------------------------------------------------------------------------#')
print(' ')

#-------------------------------------------------------------------------------------------------------------------#

# Representamos módulo, fase, diagrama de polos y ceros y retardo de fase con la analyze_sys() que tiene, entre
# otras cosas, indicadores (w0 y Q) de las singularidades en el diagrama de polos y ceros

print('-- Módulo, fase, diagrama de polos y ceros y retardo de fase --')
print(' ')

analyze_sys(t_hp, sys_name='Filtro pasa alto Chevyshev 2do orden')

#-------------------------------------------------------------------------------------------------------------------#
