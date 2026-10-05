import pandas as pd
import numpy as np
import kagglehub
import os
import matplotlib.pyplot as plt
import seaborn as sns


#LOAD DATA
def load_data():
    """Downloads the air quality dataset via kagglehub, loads it, and parses dates/years/months once."""
    path = kagglehub.dataset_download("devpassive/urban-air-quality-by-city-2019-2024")

    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    csv_file = os.path.join(path, csv_files[0])

    df = pd.read_csv(csv_file)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    return df


def inspect_data(df):
    """Prints basic info about the dataset columns and types."""
    print("\n--- DATA INFO ---")
    print(df.info())
    print("\n--- SHAPE ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")


def get_unique_cities(df):
    """Prints the list of unique cities in the dataset."""
    cities = df['city'].unique()
    print("\n--- CITIES IN DATASET ---")
    print(cities)
    print(f"Number of cities: {len(cities)}")


#DATA QUALITY

POLLUTANT_COLS = ['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co', 'o3']


def _existing_numeric_cols(df, candidates=POLLUTANT_COLS):
    """Returns only the candidate columns that actually exist in df."""
    return [c for c in candidates if c in df.columns]


def check_missing_values(df):
    """Reports missing values per column, and which cities/years are affected."""
    print("\n--- MISSING VALUES (per column) ---")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)

        print("\n--- CITIES AFFECTED BY MISSING VALUES ---")
        affected_cities = df[df.isna().any(axis=1)]['city'].value_counts()
        print(affected_cities)

        if 'year' in df.columns:
            print("\n--- YEARS AFFECTED BY MISSING VALUES ---")
            affected_years = df[df.isna().any(axis=1)]['year'].value_counts().sort_index()
            print(affected_years)


def check_duplicates(df):
    """Checks for fully duplicate rows and repeated city/date combinations."""
    print("\n--- DUPLICATE ROWS ---")
    dup_rows = df.duplicated().sum()
    print(f"Fully duplicated rows: {dup_rows}")

    if 'city' in df.columns and 'date' in df.columns:
        print("\n--- REPEATED CITY/DATE COMBINATIONS ---")
        dup_city_date = df.duplicated(subset=['city', 'date']).sum()
        print(f"Repeated city/date pairs (should usually be 0 if one row = one city-day): {dup_city_date}")


def check_invalid_values(df):
    """Flags negative pollutant concentrations, impossible dates, and other suspicious values."""
    print("\n--- INVALID VALUE CHECKS ---")

    numeric_cols = _existing_numeric_cols(df)
    for col in numeric_cols:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            print(f"Column '{col}': {negative_count} negative values found (should not be possible).")
        else:
            print(f"Column '{col}': no negative values.")

    if 'date' in df.columns:
        bad_dates = df['date'].isna().sum()
        print(f"\nUnparseable / impossible dates: {bad_dates}")
        if df['date'].notna().any():
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    if 'category' in df.columns:
        print("\n--- AQI CATEGORY VALUES (check for typos/inconsistent labels) ---")
        print(df['category'].unique())


def check_consistency(df):
    """Checks for inconsistent city naming (case, whitespace, near-duplicates)."""
    print("\n--- CITY NAME CONSISTENCY ---")
    if 'city' in df.columns:
        raw_cities = df['city'].unique()
        normalized = {c.strip().lower() for c in raw_cities}
        if len(raw_cities) != len(normalized):
            print("Possible inconsistent city naming detected (case/whitespace differences):")
            print(sorted(raw_cities))
        else:
            print("No obvious case/whitespace inconsistencies in city names.")

        print("\n--- OBSERVATION COUNT PER CITY (check equal coverage) ---")
        print(df['city'].value_counts())


def run_data_quality_report(df):
    """Runs the full data-quality section: missing values, duplicates, invalid values, consistency."""
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    check_missing_values(df)
    check_duplicates(df)
    check_invalid_values(df)
    check_consistency(df)


#DESCRIPTIVE STATISTICS

def get_descriptive_statistics(df):
    """Prints count/mean/median/min/max/std/quartiles/IQR for numeric columns."""
    print("\n--- DESCRIPTIVE STATISTICS (all numeric columns) ---")
    numeric_cols = _existing_numeric_cols(df)
    stats = df[numeric_cols].describe().T
    stats['median'] = df[numeric_cols].median()
    stats['iqr'] = stats['75%'] - stats['25%']
    stats = stats[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max', 'iqr']]
    print(stats)
    return stats


def get_variability_by_city(df, metric='aqi'):
    """Calculates mean/median/std/IQR/min/max per city to identify the most variable cities."""
    print(f"\n--- VARIABILITY BY CITY ({metric}) ---")

    def iqr(series):
        return series.quantile(0.75) - series.quantile(0.25)

    city_stats = df.groupby('city')[metric].agg(
        mean='mean',
        median='median',
        std='std',
        min='min',
        max='max',
        iqr=iqr,
        count='count'
    ).sort_values('std', ascending=False)

    print(city_stats)
    print(f"\nMost variable city (by std): {city_stats.index[0]}")
    return city_stats


#DISTRIBUTION ANALYSIS

def plot_distributions(df):
    """Plots histograms and an overall boxplot for each numeric pollutant/AQI column."""
    numeric_cols = _existing_numeric_cols(df)
    if not numeric_cols:
        print("No known pollutant/AQI columns found for distribution analysis.")
        return

    n = len(numeric_cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    sns.set_theme(style="whitegrid")

    # Histograms
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='steelblue')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

    # Boxplots (spread + outliers at a glance)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[numeric_cols], ax=ax, palette='Set3')
    ax.set_title('Boxplots of Pollutant / AQI Distributions')
    ax.set_ylabel('Value')
    plt.tight_layout()
    plt.show()


#SCATTER PLOT / RELATIONSHIP ANALYSIS

def plot_pollutant_relationships(df, pairs=None):
    """Plots scatter plots for given pollutant pairs (default: each pollutant vs AQI)."""
    numeric_cols = _existing_numeric_cols(df)

    if pairs is None:
        pairs = [(col, 'aqi') for col in numeric_cols if col != 'aqi' and 'aqi' in numeric_cols]

    if not pairs:
        print("Not enough numeric columns available for scatter plot analysis.")
        return

    n = len(pairs)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (x_col, y_col) in enumerate(pairs):
        sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.4, ax=axes[i])
        corr_val = df[[x_col, y_col]].corr().iloc[0, 1]
        axes[i].set_title(f'{x_col} vs {y_col} (corr: {corr_val:.2f})')

    for j in range(len(pairs), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


#OUTLIER INVESTIGATION

def investigate_outliers(df, column='aqi', k=1.5):
    """Flags outliers in `column` using the IQR method and prints them with surrounding context."""
    print(f"\n--- OUTLIER INVESTIGATION ({column}, IQR method, k={k}) ---")

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Number of outlier observations: {len(outliers)} ({len(outliers) / len(df):.2%} of rows)")

    if not outliers.empty:
        cols_to_show = [c for c in ['city', 'date', column] if c in df.columns]
        print("\nSample outliers:")
        print(outliers[cols_to_show].sort_values(column, ascending=False).head(10))

        print("\nOutliers by city:")
        print(outliers['city'].value_counts())

    return outliers


#CORRELATION (existing)

def check_correlation(df):
    """Calculates and prints the correlation matrix for pollutant/AQI columns only
    (excludes year/month, which are not meaningful pollutant correlates)."""
    numeric_cols = _existing_numeric_cols(df)
    correlation_matrix = df[numeric_cols].corr()
    print("\n--- CORRELATION MATRIX ---")
    print(correlation_matrix)


#CITY / TIME ANALYSIS (existing)

def get_average_aqi_by_city(df):
    """Calculates and prints the average AQI for each city."""
    average_aqi = df.groupby('city')['aqi'].mean().reset_index()
    print("\n--- AVERAGE AQI BY CITY ---")
    print(average_aqi)


def get_worst_air_quality_record(df):
    """Finds and prints the record with the worst air quality (highest AQI)."""
    worst_record = df.loc[df['aqi'].idxmax()]
    print("\n--- WORST AIR QUALITY RECORD ---")
    print(worst_record)


def analyze_yearly_trends(df):
    """Analyzes and prints the yearly trend of AQI for each city."""
    yearly_trends = df.groupby(['city', 'year'])['aqi'].mean().reset_index()
    print("\n--- YEARLY TREND OF AQI BY CITY ---")
    print(yearly_trends)


def aqi_category_breakdown(df):
    """Calculates and prints the breakdown of AQI categories for each city."""
    print("\n--- AQI CATEGORY BREAKDOWN ---")
    counts = df['category'].value_counts()
    print(counts)


#VISUALIZATIONS 

def filter_export_and_visualize_top5(df):
    """Filters the top 5 most polluted cities, exports their yearly trends to CSV, and plots them."""
    top_5_cities = df.groupby('city')['aqi'].mean().nlargest(5).index
    print("\n--- TOP 5 MOST POLLUTED CITIES ---")
    print(list(top_5_cities))

    df_top5 = df[df['city'].isin(top_5_cities)]
    yearly_trends = df_top5.groupby(['city', 'year'])['aqi'].mean().reset_index()

    output_filename = 'top_5_polluted_cities_yearly_trends.csv'
    yearly_trends.to_csv(output_filename, index=False)
    print(f"\nSuccessfully exported yearly trends to '{output_filename}'")

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.lineplot(
        data=yearly_trends,
        x='year',
        y='aqi',
        hue='city',
        marker='o',
        linewidth=2.5,
        palette='flare'
    )
    plt.title('Yearly AQI Trends for Top 5 Most Polluted Cities (2019-2024)', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average AQI', fontsize=12)
    plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def generate_air_quality_dashboard(df):
    """Generates a multi-panel dashboard of visualizations to answer comprehensive AQI questions."""
    _, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.set_theme(style="whitegrid")

    top_cities = df.groupby('city')['aqi'].mean().nlargest(5).reset_index()
    sns.barplot(
        data=top_cities,
        x='aqi',
        y='city',
        hue='city',
        palette='Reds_r',
        legend=False,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('1. Top 5 Most Polluted Cities (Average AQI)')
    axes[0, 0].set_xlabel('Average AQI')
    axes[0, 0].set_ylabel('City')

    df_top5 = df[df['city'].isin(top_cities['city'])]
    yearly_trends = df_top5.groupby(['city', 'year'])['aqi'].mean().reset_index()
    sns.lineplot(data=yearly_trends, x='year', y='aqi', hue='city', marker='o', linewidth=2.5, ax=axes[0, 1])
    axes[0, 1].set_title('2. Are Patterns Changing? (Yearly AQI Trends)')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Average AQI')

    sns.boxplot(data=df_top5, x='month', y='aqi', hue='city', palette='Set2', ax=axes[1, 0])
    axes[1, 0].set_title('3. When Does Pollution Peak? (Monthly Distribution)')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('AQI')
    axes[1, 0].legend(loc='upper right', fontsize='small')

    existing_cols = _existing_numeric_cols(df)
    corr = df[existing_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axes[1, 1])
    axes[1, 1].set_title('4. Pollutant Contributors (Correlation with AQI)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data()
    print("Data loaded successfully!")
    print(df.head())

    # Understand dataset
    inspect_data(df)
    get_unique_cities(df)

    # Data quality
    run_data_quality_report(df)

    # Descriptive statistics
    get_descriptive_statistics(df)
    get_variability_by_city(df, metric='aqi')

    # Distribution analysis
    plot_distributions(df)

    # City / time analysis
    get_average_aqi_by_city(df)
    get_worst_air_quality_record(df)
    analyze_yearly_trends(df)
    aqi_category_breakdown(df)

    # Pollutant relationships
    check_correlation(df)
    plot_pollutant_relationships(df)

    # Outlier investigation
    investigate_outliers(df, column='aqi')

    # Visualization / dashboard
    filter_export_and_visualize_top5(df)
    generate_air_quality_dashboard(df)