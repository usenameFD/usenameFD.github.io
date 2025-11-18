import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

def interpolation_spline_swap(df):
    df_swap = df[df["Instru"] == 'SWAP']
    # Appliquons une interpolation spline cubique
    cs = CubicSpline(df_swap['MAT'].values, df_swap['MKT'].values)

    # générons les points pour l'affichage de la courbe 
    maturities_gen = np.linspace(min(df_swap['MAT'].values), max(df_swap['MAT'].values), 100)

    swap_rate_gen = cs(maturities_gen)
    data_gen = {
        'Instru': ['SWAP'] * len(maturities_gen),
        'MAT': maturities_gen.tolist(),
        'MKT': swap_rate_gen.tolist()
    }

    data_gen = pd.DataFrame(data_gen)
    return data_gen


### Reconstitution de la courbe de taux zero coupons 

## Taux courts : MM

def interest_rate_mm( L0_T_T_plus_delta, T_delta=0.25,B_O_T =1,T=0):
  B_t_T_plus_delta = (B_O_T)/(T_delta*L0_T_T_plus_delta +1)
  R_0_T = -np.log(B_t_T_plus_delta)/(T + T_delta)
  return R_0_T, B_t_T_plus_delta


def compute_R_B (df):
    rates = []
    Bts= []
    df_mm = df[df["Instru"] == "MM"]

    # Extraire les valeurs sous forme de listes
    mat_mm = df_mm["MAT"].tolist()
    mkt_mm = df_mm["MKT"].tolist()
    for i in range(4):
        if i==0:
            rate, Bt = interest_rate_mm(mkt_mm[0])
            rates.append(rate)
            Bts.append(Bt)
        else:
            rate, Bt = interest_rate_mm( L0_T_T_plus_delta =mkt_mm[i], T_delta=mat_mm[i], B_O_T =1)
            rates.append(rate)
            Bts.append(Bt)
    return rates,Bts 
    

## Les taux d'intérêts à moyen terme 

def compute_R_B_future(df):
    df_FUT = df[df["Instru"] == "FUT"]

    # Extraire les valeurs sous forme de listes
    mat_FUT = df_FUT["MAT"].tolist()
    mkt_FUT = df_FUT["MKT"].tolist()
    mkt_FUT_adjusted = [1 - x for x in mkt_FUT]

    # Calcul des taux zero coupons associés
    Bts = [0.9775371388289023]
    rates_fut = []
    Bts_fut= []
    
    for i in range(len(mat_FUT)):
        if i==0:
            rate_fut, Bt = interest_rate_mm( L0_T_T_plus_delta = mkt_FUT_adjusted[0], T_delta=0.25,B_O_T =Bts[-1],T=1)
            rates_fut.append(rate_fut)
            Bts_fut.append(Bt)
        else:
            rate_fut, Bt = interest_rate_mm( L0_T_T_plus_delta =mkt_FUT_adjusted[i], T_delta=0.25,B_O_T =Bt, T= mat_FUT[i-1])
            rates_fut.append(rate_fut)
            Bts_fut.append(Bt)

    return rates_fut, Bts_fut

### Pour les SWAP 

def interpolation_spline_swap_1_30(df):
    df_swap = df[df["Instru"] == 'SWAP']
    # Appliquons une interpolation spline cubique
    cs = CubicSpline(df_swap['MAT'].values, df_swap['MKT'].values)

    # générons les points pour l'affichage de la courbe 
    maturities_gen = np.arange(3, 31, 1)

    swap_rate_gen = cs(maturities_gen)
    data_gen = {
        'Instru': ['SWAP'] * len(maturities_gen),
        'MAT': maturities_gen.tolist(),
        'MKT': swap_rate_gen.tolist()
    }

    data_gen = pd.DataFrame(data_gen)
    return data_gen


def rate_for_swap(s_swap,  cumul, B_0_0 =1, T_swap = 3):
    B_0_T = ((B_0_0 -(s_swap)*cumul))/(1+s_swap)
    cumul = cumul + B_0_T
    rate = -np.log(B_0_T)/T_swap
    return B_0_T, cumul,rate

def compute_R_B_swap(maturities_selected, rates_selected, Bts_fut_init, Bts_init):
    Bts_swap = []
    rates_swap = []
    cumul = Bts_fut_init + Bts_init
    for i in range(len(maturities_selected)):
        B_0_T, cumul,rate = rate_for_swap(s_swap=rates_selected[i], B_0_0 =1, cumul = cumul, T_swap = maturities_selected[i])
        Bts_swap.append(B_0_T)
        rates_swap.append(rate)
    return Bts_swap, rates_swap