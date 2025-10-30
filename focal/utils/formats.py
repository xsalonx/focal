def sci_e(x, prec=6, keep_plus=True):
    if x == '':
        return ''
    s = f"{x:.{prec}e}"           # e.g. '2.800000e-05'
    m, exp = s.split('e', 1)
    m = m.rstrip('0').rstrip('.') # '2.8'
    sign, digits = exp[0], exp[1:]
    digits = digits.lstrip('0') or '0'  # '05' -> '5', '00' -> '0'
    if not keep_plus and sign == '+':
        exp_str = digits
    else:
        exp_str = sign + digits
    return f"{m}e{exp_str}"
