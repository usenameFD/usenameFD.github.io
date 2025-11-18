import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal

# Summarize catégorical data

def describe_cat(df):
    for col in df.columns:
        print(f"\nProportion de chaque catégorie dans {col}:")
        print(df[col].value_counts(normalize=True))
        print('_._._._._._._._._._._._._._._._._._._.__._._')


def sampleData(df, var, frac, rand_state=49):
    # Création de la colonne de stratification combinée
    df['strat_var'] = df[var].astype(str).agg('_'.join, axis=1)

    # Utilisation de train_test_split pour tirer un échantillon stratifié
    train, test = train_test_split(
        df,
        test_size=frac,
        stratify=df['strat_var'],
        random_state=rand_state
    )
    # Suppression de la colonne intermédiaire
    train = train.drop(columns='strat_var')
    test = test.drop(columns='strat_var')
    
    return train, test

def biva_quali(data, var1, var2):
    # Calcul des proportions
    proportions = pd.crosstab(data[var1], data[var2], normalize='index')

    # Création du graphique
    ax = proportions.plot(kind='bar', stacked=True, figsize=(10, 6))

    # Ajouter des labels et un titre
    plt.title(f'Répartition de {var2} par catégorie de {var1}')
    plt.xlabel(f'Catégories de {var1}')
    plt.ylabel('Proportion')

    # Ajustement des étiquettes en abscisse
    plt.xticks(
        ticks=range(0, len(proportions.index), max(1, len(proportions.index) // 10)),  # Afficher 1 label sur 10 (ajustable)
        labels=proportions.index[::max(1, len(proportions.index) // 10)],  # Sélectionner ces labels
        rotation=30,  # Rotation des labels
        ha='right',  # Alignement à droite
        fontsize=10  # Réduction de la taille du texte
    )

    plt.legend(title=var2)
    plt.tight_layout()  # Ajustement des marges pour éviter les labels coupés

    plt.show()


def selectFeatures(data, col_num, target, method='pearson'):
    """
    Select features based on Kruskal-Wallis test and correlation threshold.

    Parameters:
    - data: DataFrame containing the dataset
    - target: Target variable (categorical column name)
    - method: Correlation method ('spearman', 'pearson', or 'kendall')

    Returns:
    - feat: List of selected feature names
    """
    # Step 1: Perform Kruskal-Wallis test for each numerical feature
    results = {}
       
    for col in col_num:
        grouped_data = [data[data[target] == g][col] for g in data[target].unique()]
        stat, p_value = kruskal(*grouped_data)
        results[col] = {'stat': stat, 'p_value': p_value}

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    results_df['significant'] = results_df['p_value'] < 0.05
    results_df = results_df[results_df['significant']].sort_values('stat', ascending=False)
    
    # If no features are significant, return an empty list
    if results_df.empty:
        return []

    # Step 2: Calculate correlation matrix
    corr_matrix = data[col_num].corr(method=method)

    # Step 3: Select features based on Kruskal-Wallis stat and correlation threshold
    feat = [results_df['stat'].idxmax()]  # Start with the most significant feature
    results_df = results_df.drop(feat[0])  # Drop the selected feature

    while not results_df.empty:
        col = results_df['stat'].idxmax()  # Most significant remaining feature
        results_df = results_df.drop(col)  # Remove it from results
        is_valid = True
        
        # Check correlation with already selected features
        for valid in feat:
            if abs(corr_matrix.loc[col, valid]) > 0.6:
                is_valid = False
                break
        
        if is_valid:
            feat.append(col)

    return feat



def stabilite_structurelle(data, var, datdelhis, title):
    df = data.copy()
        
    # proportion mensuelle par catégorie
    df = data.groupby([datdelhis, var]).size().reset_index(name='count')
    df['freq'] = df.groupby([datdelhis])['count'].transform(lambda x: x / x.sum()) * 100
    
    # Evolution mensuelle 
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=datdelhis, y='freq', hue=var, marker='o')
    plt.title(title)
    plt.xlabel("Date d'observation")
    plt.ylabel('Pourcentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return df

def concentration_plot(data, hcr, target, date_column, label="HCR concentration through issue date", save_figures=False, output_dir="./"):
    """
    Plots volume concentration and default rate over time, with an option to save the figures.

    Parameters:
    data (pd.DataFrame): Input DataFrame.
    hcr (str): Column name for the risk class (e.g., 'CHR').
    target (str): Target column (e.g., default indicator).
    date_column (str): Column containing date/time information.
    label (str): Plot label.
    save_figures (bool): Whether to save the figures.
    output_dir (str): Directory to save the figures.
    """
    df_score = data.copy()

    # Ensure the date column is in datetime format
    df_score[date_column] = pd.to_datetime(df_score[date_column].astype(str) + '-01-01')
    
    # Group by year and category, then calculate proportions
    df_score = df_score.groupby([pd.Grouper(key=date_column, freq='Y'), hcr]).agg(
        part_defaut=(target, 'sum'), 
        volume=(target, 'count')
    ).reset_index()
    
    # Compute correct default rate per risk class per year
    df_score['part_defaut'] = df_score['part_defaut'] * 100 / df_score['volume']

    # Compute correct volume share per year
    df_score['volume'] = df_score.groupby(date_column)['volume'].transform(lambda x: x / x.sum()) * 100

    # Handle NaNs (if any)
    df_score.fillna(0, inplace=True)

    # First Figure: Improved Volume Concentration (Combined)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_score, x=df_score[date_column].dt.strftime('%Y'), y='volume', hue=hcr, ax=ax1, palette='coolwarm')

    ax1.set_xlabel('Issue Year')
    ax1.set_ylabel('Concentration (%)')
    ax1.set_title(f'Volume by {date_column} over years')
    ax1.legend(title=hcr, bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside the plot

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the first figure if requested
    if save_figures:
        output_path = f"{output_dir}volume_concentration_combined.png"
        fig1.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved combined volume concentration plot to {output_path}")

    plt.show()

    # Separate Figure for Each Risk Class (HCR)
    for cl_chr in df_score[hcr].unique():
        fig, ax1 = plt.subplots(figsize=(12, 6))  # Create a new figure for each risk class
        seg_score = df_score[df_score[hcr] == cl_chr]  # Filter data for this class
        
        # Improved Bar Plot (Concentration)
        sns.barplot(data=seg_score, x=seg_score[date_column].dt.strftime('%Y'), y='volume', ax=ax1, color='blue', alpha=0.6, label='Concentration')

        # Improved Line Plot for Default Rate (on secondary y-axis)
        ax2 = ax1.twinx()  # Create a secondary y-axis for default rate
        sns.lineplot(data=seg_score, x=seg_score[date_column].dt.strftime('%Y'), y='part_defaut', ax=ax2, color='red', marker='o', linestyle='-', linewidth=2, label='Default Rate')
        
        # Add labels to the bars (Volume)
        for i, (date, value) in enumerate(zip(seg_score[date_column].dt.strftime('%Y'), seg_score.volume)):
            ax1.text(i, value, f'{value:.1f}%', color='blue', ha='center', va='bottom')

        # Add labels to the points (Default Rate)
        for i, (date, value) in enumerate(zip(seg_score[date_column].dt.strftime('%Y'), seg_score.part_defaut)):
            ax2.text(i, value, f'{value:.1f}%', color='red', ha='center', va='bottom')

        # Labels & Title
        ax1.set_xlabel('Issue Year')
        ax1.set_ylabel('Concentration (%)')
        ax2.set_ylabel('Default Rate (%)')
        ax1.set_title(f'Volume and Default Rate for HRC_{cl_chr} through {date_column}')

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the individual figure if requested
        if save_figures:
            output_path = f"{output_dir}volume_default_rate_HRC_{cl_chr}.png"
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Saved plot for HRC_{cl_chr} to {output_path}")

        plt.show()




def ecart_relatif(taux1, taux2):
    if taux2 != 0:
        return ((taux2 - taux1) / taux2) * 100
    

def taux_defaut(data, var, date_column, target):
    """
    Calculates and plots the default rate over time (yearly aggregation) for categories of a variable.

    Parameters:
    - data: DataFrame containing the data.
    - var: Name of the categorical variable to analyze.
    - date_column: Name of the date column for time-based grouping.
    - target: Name of the target variable (e.g., default indicator).

    Returns:
    - DataFrame containing default rates.
    - Displays a line plot of default rates over time (yearly).
    """
    # Extract the year from the date_column
    data['year'] = pd.to_datetime(data[date_column]).dt.year

    # Group by year, var, and target to calculate counts
    df = data.groupby(['year', var, target]).size().reset_index(name='count')
    
    # Calculate the default rate for each year and category
    df['taux_defaut'] = df.groupby(['year', var])['count'].transform(lambda x: x / x.sum()) * 100
    
    # Filter for default cases (target == 1)
    df_defaut = df[df[target] == 1]
    
    # Plot the graph
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_defaut, x='year', y='taux_defaut', hue=var)  # Use 'year' for x-axis
    plt.title(f'HCR Default Rate through years')
    plt.xlabel("Date")
    plt.ylabel('Default Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Calculate relative differences between categories
    distribution = data.groupby("CHR")[target].mean()
    distribution = distribution.sort_values()
    
    differences_relatives = distribution.pct_change() * 100
    
    # Créer le DataFrame avec les résultats
    resultats = pd.DataFrame({
        "Default rate": distribution,
        "Relatif deviation": differences_relatives
    })
    resultats["Gini index"] = 2*resultats["Default rate"]*(1-resultats["Default rate"])
    resultats["Shannon entropy"] = -resultats["Default rate"]*np.log(resultats["Default rate"])\
                                    - (1-resultats["Default rate"])*np.log(1-resultats["Default rate"])
    
    resultats = resultats.sort_values(by="Default rate", ascending=True)
   
  
    return resultats



def affiche_ecart_relatif(data, var):
    """
    Affiche l'écart relatif moyen entre les taux de défaut moyens des catégories d'une variable.
    """
    # Calcul des taux de défaut
    taux_data = taux_defaut(data, var)
    
    # Calcul des moyennes des taux de défaut par catégorie
    mean_taux = taux_data.groupby(var)['taux_defaut'].mean()
    
    # Vérification qu'il y a au moins deux catégories à comparer
    if len(mean_taux) > 1:
        ecart = ecart_relatif(mean_taux.iloc[0], mean_taux.iloc[1])
        print(f"L'écart relatif entre les taux de défaut moyens des catégories '{mean_taux.index[0]}' et '{mean_taux.index[1]}': {ecart:.2f}%")
    else:
        print("Pas assez de catégories pour calculer un écart relatif.")




# Outil de discretisation
def discretizer(data, feat, target, max_depth=2):
    from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

    # Fit a DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, min_samples_leaf=0.05, random_state=49)
    clf.fit(data[feat].to_numpy().reshape(-1, 1), data[target].to_numpy().reshape(-1, 1))

    # Export tree structure as text
    tree_text = export_text(clf, feature_names=[feat])

    # Initialize conditions and values
    conditions = []
    thresholds = set()

    # Parse tree text to extract thresholds
    for line in tree_text.split("\n"):
        line = line.strip()  # Remove leading/trailing whitespace
        if "<=" in line or ">" in line:  # Look for valid condition lines
            parts = line.split()
            for part in parts:
                try:
                    # Try to parse the threshold as a float
                    threshold = float(part)
                    thresholds.add(threshold)
                except ValueError:
                    # Skip parts that are not numbers
                    continue

    # Sort thresholds to build conditions
    thresholds = sorted(thresholds)
    conditions.append((data[feat] <= thresholds[0]))  # First condition
    for i in range(1, len(thresholds)):
        conditions.append((data[feat] > thresholds[i - 1]) & (data[feat] <= thresholds[i]))
    conditions.append((data[feat] > thresholds[-1]))  # Last condition

    # Assign values for each condition
    values = list(range(1, len(conditions) + 1))


    # Visualize 
    class_names = [str(cls) for cls in clf.classes_]

    plt.figure(figsize=(12, 8))
    plot_tree(
    clf, 
    feature_names=[feat],  # Replace with your actual feature names
    class_names=class_names, 
    filled=True
    )
    plt.title("Decision Tree Visualization")
    plt.show()

    return conditions, values, thresholds


# Outil de discretisation
def discretizer(data, feat, target, max_depth=2, min_samples_leaf=0.05, criterion='gini'):
    from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

    # Gerer les outliers avant discrétisation
    def no_outliers(data,feat):
        Q1 = data[feat].quantile(0.05)
        Q3 = data[feat].quantile(0.95)
        IQ = Q3 - Q1
        b_inf = Q1 - 1.5 * IQ
        b_sup = Q3 + 1.5 * IQ

        # Clip the feature values
        clipped_feat = data[feat].clip(lower=b_inf, upper=b_sup)
    
        df = pd.DataFrame({
            feat: clipped_feat,
            target: data[target]
        })
        return df, b_inf, b_sup

    # Appliquer le traitements des outliers
    df_clean = no_outliers(data,feat)
    df = df_clean[0]
    
    # Fit a DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=0.05, random_state=49)
    clf.fit(df[feat].to_numpy().reshape(-1, 1), df[target].to_numpy().reshape(-1, 1))

    # Export tree structure as text
    tree_text = export_text(clf, feature_names=[feat])

    # Initialize conditions and values
    conditions = []
    thresholds = set()

    # Parse tree text to extract thresholds
    for line in tree_text.split("\n"):
        line = line.strip()  # Remove leading/trailing whitespace
        if "<=" in line or ">" in line:  # Look for valid condition lines
            parts = line.split()
            for part in parts:
                try:
                    # Try to parse the threshold as a float
                    threshold = float(part)
                    thresholds.add(threshold)
                except ValueError:
                    # Skip parts that are not numbers
                    continue

    # Sort thresholds to build conditions
    thresholds = sorted(thresholds)
    conditions.append((data[feat] <= thresholds[0]))  # First condition
    for i in range(1, len(thresholds)):
        conditions.append((data[feat] > thresholds[i - 1]) & (data[feat] <= thresholds[i]))
    conditions.append((data[feat] > thresholds[-1]))  # Last condition

    # Assign values for each condition
    values = list(range(1, len(conditions) + 1))
    values.reverse()


    # Visualize 
    class_names = [str(cls) for cls in clf.classes_]

    plt.figure(figsize=(12, 8))
    plot_tree(
    clf, 
    feature_names=[feat],  # Replace with your actual feature names
    class_names=class_names, 
    filled=True
    )
    plt.title("Decision Tree Visualization")
    plt.show()

    # Ajoute des bornes intercaltiles aux seuils
    thresholds.append(df_clean[1]) # Borne inf
    thresholds.append(df_clean[2]) # Borne sup

    return conditions, values, sorted(thresholds)



# Représentation des classes de risque
def CHR_plot(df_score, target, label='train'):
    seg_score = df_score.groupby(['CHR', target]).agg(part_defaut=(target,'sum'), volume=(target,'count')).reset_index()
    seg_score = seg_score.groupby('CHR').agg({'part_defaut':'sum','volume':'sum'}).reset_index()
    seg_score['part_defaut'] = seg_score['part_defaut']  / seg_score['volume'] 
    seg_score['volume'] = seg_score['volume']  / seg_score['volume'].sum()


    # Volume
    plt.bar(seg_score.CHR, seg_score.volume, color='blue', alpha=0.7, label='Volume proportion')

    # Part défaut
    plt.plot(seg_score.CHR, seg_score.part_defaut, color='red', marker='o', linestyle='-', linewidth=2, label='Default Rate')

    # # Add labels to the points (Gini values)
    for i, value in enumerate(seg_score.volume):
        plt.text(seg_score.CHR[i], value, f'{value*100:.1f}%', color='blue', ha='center', va='bottom')

    # Add labels to the points (Variation of Gini values)
    for i, value in enumerate(seg_score.part_defaut):
        plt.text(seg_score.CHR[i], value, f'{value*100:.1f}%', color='red', ha='center', va='top')

    # Add labels and title
    plt.xlabel('HCR')
    plt.ylabel('Volume / Default rate')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Volumetry and Defaulft rate across HCR ({label} set)')
    plt.legend()  # Add legend to explain the bars and curve

    # Show the plot
    plt.tight_layout()  # Adjust layout for better fit
    plt.show()



def biva_quali(data, var1, var2):
    # Calcul des proportions
    proportions = pd.crosstab(data[var1], data[var2], normalize='index')

    # Création du graphique
    ax = proportions.plot(kind='bar', stacked=True, figsize=(10, 6))

    # Ajouter des labels et un titre
    plt.title(f'Répartition de {var2} par catégorie de {var1}')
    plt.xlabel(f'Catégories de {var1}')
    plt.ylabel('Proportion')

    # Ajustement des étiquettes en abscisse
    plt.xticks(
        ticks=range(0, len(proportions.index), max(1, len(proportions.index) // 10)),  # Afficher 1 label sur 10 (ajustable)
        labels=proportions.index[::max(1, len(proportions.index) // 10)],  # Sélectionner ces labels
        rotation=30,  # Rotation des labels
        ha='right',  # Alignement à droite
        fontsize=10  # Réduction de la taille du texte
    )

    plt.legend(title=var2)
    plt.tight_layout()  # Ajustement des marges pour éviter les labels coupés

    plt.show()
    

def anova_test(base, hcr=1, target_col="Score", group_col="CHR"):
    """
    Perform ANOVA test for each column in the dataset to check for homogeneity.
    
    Parameters:
        base (pd.DataFrame): The dataset containing the features and target variable.
        hcr (int): The value of the 'CHR' column to filter the data.
        target_col (str): The name of the target variable (default: "Score").
        group_col (str): The name of the group column (default: "CHR").
    
    Returns:
        pd.DataFrame: A DataFrame with ANOVA results for each feature.
    """
    anova_results = []
    
    # Filter data for the specified 'CHR' value
    filtered_data = base[base[group_col] == hcr]
    
    # Perform ANOVA for each column
    for col in filtered_data.columns:
        if col == target_col or col == group_col:  # Skip the target and group columns
            continue
        
        # Fit the model
        model = ols(f'{target_col} ~ C({col})', data=filtered_data).fit()
        
        # Perform ANOVA
        anova_result = anova_lm(model)
        f_statistic = anova_result['F'][0]
        p_value = anova_result['PR(>F)'][0]
        
        # Append results to the list
        anova_results.append({
            "Variable": col,
            "F_statistic": f_statistic,
            "p-value": p_value,
            "Decision": "Not homogeneous" if p_value < 0.05 else "Homogeneous"
        })
    
    # Convert results to a DataFrame
    anova_results_df = pd.DataFrame(anova_results)
    
    return anova_results_df

def homogeneous_feat(base, target_col="Score", group_col="CHR"):
    """
    Identify homogeneous features for each unique value of 'CHR'.
    
    Parameters:
        base (pd.DataFrame): The dataset containing the features and target variable.
        target_col (str): The name of the target variable (default: "Score").
        group_col (str): The name of the group column (default: "CHR").
    
    Returns:
        pd.DataFrame: A DataFrame summarizing homogeneous features for each 'CHR' group.
        pd.DataFrame: A DataFrame with combined ANOVA results for all 'CHR' groups.
    """
    # Initialize results
    results = []
    combined_anova_results = []
    
    # Iterate over each unique value of 'CHR'
    for hcr in base[group_col].unique():
        # Perform ANOVA for the current 'CHR' group
        anova_results_df = anova_test(base, hcr=hcr, target_col=target_col, group_col=group_col)
        
        # Extract homogeneous variables
        homogeneous_vars = anova_results_df.loc[anova_results_df['Decision'] == "Homogeneous", "Variable"].tolist()
        
        # Append results
        results.append({
            group_col: hcr,
            "Homogeneous features": homogeneous_vars,
            "Number of Homogeneous features": len(homogeneous_vars)
        })
        
        # Add 'CHR' column to ANOVA results for identification
        anova_results_df[group_col] = hcr
        combined_anova_results.append(anova_results_df)
    
    # Convert results to DataFrames
    results_df = pd.DataFrame(results)
    combined_anova_results_df = pd.concat(combined_anova_results, ignore_index=True)
    
    return results_df, combined_anova_results_df

def ks_test(data, col_chr, col_target, col_score):
    result = {"CHR": [], "statistic": [], "pvalue": []}
    CHR = list(data[col_chr].unique())

    for chr in CHR:
        base = data[data[col_chr] == chr]
        sains = base[base[col_target] == 0]
        defaut = base[base[col_target] == 1]

        score_sains = sains[col_score]
        score_defauts = defaut[col_score]

        # Test KS
        ks = ks_2samp(score_defauts, score_sains)

        # Store results
        result["CHR"].append(chr)
        result["statistic"].append(ks.statistic)
        result["pvalue"].append(ks.pvalue)

    return pd.DataFrame(result)

# Reformater les données
def formater(df, col):
    data = df[col]
    # 1: Discretize 'TAV'
    conditions = [
    (data['TAV']<= 11.18),
    ((data['TAV']> 11.18)&(data['TAV']<=11.68)),
    ((data['TAV']>11.68)&(data['TAV']<=11.95)),
    (data['TAV']> 11.18)
        ]

    values = [4,3,2,1]
    data['TAV'] = np.select(conditions, values)

    # 2: Discretize 'PN20MM'
    conditions = [
    (data['PN20MM']<=1.5),
    ((data['PN20MM']>1.5)&(data['PN20MM']<=2.5)),
    ((data['PN20MM']>2.5)&(data['PN20MM']<=8.5)),
    (data['PN20MM']>8.5),
        ]

    values = [4,3,2,1]
    data['PN20MM'] = np.select(conditions, values)

    # 3: Discretize 'PQ90'
    conditions = [
    (data['PQ90']<=5.32),
    ((data['PQ90']>5.32)&(data['PQ90']<=6.04)),
    (data['PQ90']>6.04)
        ]

    values = [1,2,3]
    data['PQ90'] = np.select(conditions, values)

    # 4: Discretize 'PQ99'
    conditions = [
    (data['PQ99']<=17.48),
    ((data['PQ99']>17.48)&(data['PQ99']<=22.7)),
    ((data['PQ99']>22.7)&(data['PQ99']<=25.37)),
    (data['PQ99']>25.37),
        ]

    values = [4,3,2,1]
    data['PQ99'] = np.select(conditions, values)

    # 5: Discretize 'RR99'
    conditions = [
    (data['RR99']<=1.5),
    ((data['RR99']>1.5)&(data['RR99']<=2.5)),
    ((data['RR99']>2.5)&(data['RR99']<=3.5)),
    (data['RR99']>3.5),
        ]

    values = [4,3,2,1]
    data['RR99'] = np.select(conditions, values)
    return data


