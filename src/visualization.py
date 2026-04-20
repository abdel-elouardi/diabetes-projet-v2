import seaborn as sns
import matplotlib.pyplot as plt

def plot_distributions(df):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], ax=axes[i], kde=True, color='steelblue')
        axes[i].set_title(col)
    plt.suptitle("Distribution des variables", fontsize=16)
    plt.tight_layout()
    plt.savefig("distributions.png")
    plt.show()
    print("✅ distributions.png sauvegardé")

def plot_correlation(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Heatmap de corrélation")
    plt.tight_layout()
    plt.savefig("correlation.png")
    plt.show()
    print("✅ correlation.png sauvegardé")

def plot_boxplots(df):
    cols = df.select_dtypes(include='number').columns.tolist()
    if 'Outcome' in cols:
        cols.remove('Outcome')
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.boxplot(x='Outcome', y=col, data=df, ax=axes[i],
                    hue='Outcome', palette='Set2', legend=False)
        axes[i].set_title(col)
    plt.suptitle("Boxplots par rapport au diabète", fontsize=16)
    plt.tight_layout()
    plt.savefig("boxplots.png")
    plt.show()
    print("✅ boxplots.png sauvegardé")

def plot_countplot(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Outcome', data=df,
                  hue='Outcome', palette='Set2', legend=False)
    plt.title("Équilibre des classes")
    plt.xticks([0, 1], ['Non diabétique', 'Diabétique'])
    plt.tight_layout()
    plt.savefig("countplot.png")
    plt.show()
    print("✅ countplot.png sauvegardé")

def plot_pairplot(df):
    sns.pairplot(df, hue='Outcome', palette='Set2')
    plt.suptitle("Pairplot", y=1.02, fontsize=16)
    plt.savefig("pairplot.png")
    plt.show()
    print("✅ pairplot.png sauvegardé")

def visualize_all(df):
    print("🎨 Génération des graphiques...\n")
    plot_distributions(df)
    plot_correlation(df)
    plot_boxplots(df)
    plot_countplot(df)
    plot_pairplot(df)
    print("\n✅ Tous les graphiques générés !")