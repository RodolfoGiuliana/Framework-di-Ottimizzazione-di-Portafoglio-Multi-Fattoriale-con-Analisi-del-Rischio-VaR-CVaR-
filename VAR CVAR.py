import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


np.random.seed(42)
# Sample assets and returns data
assets = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA']
returns = pd.DataFrame(np.random.normal(0.001, 0.02, size=(252, len(assets))), columns=assets)
opt_weights = np.array([0.2, 0.2, 0.15, 0.15, 0.15, 0.15])  # Sample weights
portfolio_rets = returns.dot(opt_weights)  # Portfolio returns
cumulative_rets = (1 + portfolio_rets).cumprod()  # Cumulative returns


fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# --- GRAFICO 1: Andamento Storico ---
axs[0, 0].plot(cumulative_rets, color='navy', lw=2)
axs[0, 0].set_title('Andamento Storico Portafoglio Ottimizzato', fontweight='bold')
axs[0, 0].grid(True, alpha=0.3)

# --- GRAFICO 2: Distribuzione Pesi (Pie Chart) ---
axs[0, 1].pie(opt_weights, labels=assets[:len(opt_weights)], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
axs[0, 1].set_title('Asset Allocation Ottimale', fontweight='bold')

# --- GRAFICO 3: Analisi del Rischio (Istogramma Rendimenti) ---
axs[1, 0].hist(portfolio_rets, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
var_95 = np.percentile(portfolio_rets, 5)
axs[1, 0].axvline(var_95, color='red', linestyle='--', label=f'VaR 95%: {var_95:.2%}')
axs[1, 0].set_title('Distribuzione Rendimenti e VaR', fontweight='bold')
axs[1, 0].legend()

# --- GRAFICO 4: Correlazione tra Asset ---
sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', ax=axs[1, 1])
axs[1, 1].set_title('Matrice di Correlazione Asset', fontweight='bold')

# Adjust layout
plt.tight_layout()
plt.show()

# Define get_portfolio_stats function
def get_portfolio_stats(weights):
    port_returns = np.sum(returns.mean() * weights) * 252  # Annualized return
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = port_returns / port_volatility
    return port_returns, port_volatility, sharpe_ratio

# 4. REPORT FINALE
stats = get_portfolio_stats(opt_weights)
print("\n" + "="*40)
print("       REPORT PORTAFOGLIO FINTECH")
print("="*40)

for i, asset in enumerate(assets[:len(opt_weights)]):
    print(f"{asset:6} | Peso: {opt_weights[i]:.2%}")
print("-" * 40)
print(f"Rendimento Annuo Atteso: {stats[0]:.2%}")
print(f"Volatilità (Rischio):    {stats[1]:.2%}")
print(f"Sharpe Ratio:           {stats[2]:.2f}")
print(f"Value at Risk (95%):    {var_95:.2%}")
print("="*40)
