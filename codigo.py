import math
import matplotlib.pyplot as plt
 
# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------
pi = math.pi
DELTA = 1e-5          # passo das diferenças finitas
V_MIN = 500.0         # volume mínimo (m³)
R_MIN = 1e-6          # evita r, h <= 0 durante as iterações
 
 
# ----------------------------------------------------------------------
# Funções do problema
# ----------------------------------------------------------------------
def custo(r, h):
    """Função objetivo f(r, h) (eq. 1)."""
    return 350 * pi * r**2 + 200 * pi * r * h
 
 
def restricao(r, h):
    """Restrição g(r, h) <= 0 (eq. 2)."""
    return V_MIN - pi * r**2 * h
 
 
def custo_penalizado(r, h, mu):
    """Função auxiliar P(r, h) = f + mu * max(0, g)^2 (eq. 3)."""
    return custo(r, h) + mu * max(0.0, restricao(r, h)) ** 2
 
 
# ----------------------------------------------------------------------
# Gradiente numérico
# ----------------------------------------------------------------------
def gradiente(r, h, mu, delta=DELTA):
    """Gradiente de P por diferenças finitas adiantadas (eqs. 4 e 5)."""
    P0 = custo_penalizado(r, h, mu)
    dP_dr = (custo_penalizado(r + delta, h, mu) - P0) / delta
    dP_dh = (custo_penalizado(r, h + delta, mu) - P0) / delta
    return dP_dr, dP_dh
 
 
# ----------------------------------------------------------------------
# Método do gradiente
# ----------------------------------------------------------------------
def gradiente_descendente(r0, h0, alpha, mu, max_iter=1_000_000,
                          tol=1.0, salvar_a_cada=100):
    """Minimiza P(r, h) com r <- r - alpha*dP/dr e h <- h - alpha*dP/dh
 
    Parada: norma do gradiente < tol (convergiu), divergência (nan/inf ou
    valores absurdos) ou max_iter atingido
 
    Retorna um dicionário com resultado final e histórico (amostrado a cada
    `salvar_a_cada` iterações, para não gastar memória)
    """
    r, h = r0, h0
    hist = {"k": [], "r": [], "h": [], "P": [], "custo": [], "norma": []}
    convergiu = False
    divergiu = False
    k = 0
 
    def registrar(k, r, h, norma):
        hist["k"].append(k)
        hist["r"].append(r)
        hist["h"].append(h)
        hist["P"].append(custo_penalizado(r, h, mu))
        hist["custo"].append(custo(r, h))
        hist["norma"].append(norma)
 
    for k in range(max_iter):
        dP_dr, dP_dh = gradiente(r, h, mu)
        norma = math.hypot(dP_dr, dP_dh)
 
        if k % salvar_a_cada == 0:
            registrar(k, r, h, norma)
 
        if norma < tol:
            convergiu = True
            break
 
        r = max(r - alpha * dP_dr, R_MIN)
        h = max(h - alpha * dP_dh, R_MIN)
 
        if not (math.isfinite(r) and math.isfinite(h)) or r > 1e6 or h > 1e6:
            divergiu = True
            break
    else:
        k = max_iter
 
    norma = math.hypot(*gradiente(r, h, mu)) if not divergiu else float("nan")
    registrar(k, r, h, norma)
    return {"r": r, "h": h, "iter": k, "convergiu": convergiu,
            "divergiu": divergiu, "hist": hist}
 
 
# ----------------------------------------------------------------------
# Referência analítica (para validação)
# Com h = V_MIN/(pi r²): f(r) = 350 pi r² + 200*V_MIN/r
# f'(r) = 0  ->  r³ = 200*V_MIN / (700 pi)
# ----------------------------------------------------------------------
def otimo_analitico():
    r = (200 * V_MIN / (700 * pi)) ** (1 / 3)
    h = V_MIN / (pi * r**2)
    return r, h, custo(r, h)
 
 
def status(res):
    if res["divergiu"]:
        return "divergiu"
    return "convergiu" if res["convergiu"] else "max_iter"
 
 
def executar(r0, h0, alpha, mu, **kw):
    res = gradiente_descendente(r0, h0, alpha, mu, **kw)
    r, h = res["r"], res["h"]
    res["volume"] = pi * r**2 * h
    res["custo_final"] = custo(r, h)
    return res
 
 
# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    R0, H0 = 5.0, 10.0          # chute inicial (viável)
    r_ref, h_ref, c_ref = otimo_analitico()
    print(f"Ótimo analítico: r = {r_ref:.4f}, h = {h_ref:.4f}, "
          f"custo = {c_ref:.2f}\n")
 
    # --- Experimento 1: variação de mu, com alpha estável para cada mu ---
    # Estabilidade aproximada: alpha < 2 / (2*mu*|grad g|²) ~ 1.25e-5 / mu
    casos_mu = [(1e1, 1e-6), (1e2, 1e-7), (1e3, 1e-8), (1e4, 1e-9)]
    # --- Experimento 2: variação de alpha, com mu = 1e2 fixo ---
    casos_alpha = [(1e2, 1e-5), (1e2, 1e-6), (1e2, 1e-7), (1e2, 1e-8)]
 
    resultados = {}
    print(f"{'mu':>8} {'alpha':>8} {'r':>9} {'h':>9} {'volume':>9} "
          f"{'custo':>10} {'iter':>9}  status")
    for titulo, casos in (("Variação de mu", casos_mu),
                          ("Variação de alpha", casos_alpha)):
        print(f"--- {titulo} ---")
        for mu, alpha in casos:
            if (mu, alpha) not in resultados:
                resultados[(mu, alpha)] = executar(R0, H0, alpha, mu)
            res = resultados[(mu, alpha)]
            print(f"{mu:8.0e} {alpha:8.0e} {res['r']:9.4f} {res['h']:9.4f} "
                  f"{res['volume']:9.2f} {res['custo_final']:10.1f} "
                  f"{res['iter']:9d}  {status(res)}")
 
    # --- Gráficos ---
    # 1) convergência do custo para cada mu (alpha estável)
    plt.figure(figsize=(7, 4))
    for mu, alpha in casos_mu:
        h_ = resultados[(mu, alpha)]["hist"]
        plt.plot(h_["k"], h_["custo"], label=f"mu={mu:.0e}, alpha={alpha:.0e}")
    plt.axhline(c_ref, color="k", ls="--", lw=0.8, label="ótimo analítico")
    plt.xscale("log")
    plt.xlabel("iteração")
    plt.ylabel("custo f(r, h) [R$]")
    plt.title("Convergência do custo para diferentes mu")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergencia_custo.png", dpi=150)
 
    # 2) trajetória de r e h (mu = 1e2, alpha = 1e-7)
    h_ = resultados[(1e2, 1e-7)]["hist"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(h_["k"], h_["r"], label="r")
    ax[0].plot(h_["k"], h_["h"], label="h")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("iteração")
    ax[0].set_ylabel("metros")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[0].set_title("r e h por iteração")
    ax[1].plot(h_["k"], h_["norma"])
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("iteração")
    ax[1].set_ylabel("||grad P||")
    ax[1].grid(alpha=0.3)
    ax[1].set_title("Norma do gradiente")
    plt.tight_layout()
    plt.savefig("trajetoria.png", dpi=150)
 
    # 3) curvas de nível de P com a trajetória (mu = 1e2)
    mu = 1e2
    rs = [1.5 + 0.05 * i for i in range(0, 80)]
    hs = [4 + 0.15 * j for j in range(0, 80)]
    Z = [[custo_penalizado(r, h, mu) for r in rs] for h in hs]
    plt.figure(figsize=(6, 5))
    cs = plt.contour(rs, hs, Z, levels=[4.3e4, 4.6e4, 5e4, 6e4, 8e4, 1.2e5])
    plt.clabel(cs, fontsize=7)
    plt.plot(h_["r"], h_["h"], "r.-", ms=3, lw=0.8, label="trajetória")
    plt.plot([r_ref], [h_ref], "k*", ms=10, label="ótimo analítico")
    plt.xlabel("r [m]")
    plt.ylabel("h [m]")
    plt.legend()
    plt.title("Curvas de nível de P (mu=1e2)")
    plt.tight_layout()
    plt.savefig("curvas_nivel.png", dpi=150)
    print("\nGráficos salvos: convergencia_custo.png, trajetoria.png, "
          "curvas_nivel.png")
