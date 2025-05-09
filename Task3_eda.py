import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import networkx as nx
from itertools import combinations
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def top_project_rdc():
    top_rdc = df['projectrdc'].value_counts().head(10)
    
    fig, ax = plt.subplots()
    ax.bar(top_rdc.index, top_rdc.values)
    ax.set_xlabel('Project RDC')
    ax.set_ylabel('Number of Outputs')
    ax.set_title('Top 10 Project RDCs')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def publications_by_year():
    year_counts = df['outputyear'].value_counts().sort_index()

    fig, ax = plt.subplots()
    ax.bar(year_counts.index.astype(int), year_counts.values)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Publications')
    ax.set_title('Publications by Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def top_pi():
    project_pi = df['projectpi'].dropna().str.split(';')

    pi_list = [pi.strip() for sublist in project_pi for pi in sublist]
    counter = Counter(pi_list)
    top_pis = counter.most_common(10)
    
    # Print the top 10 authors
    print("Top 10 authors and publication counts:")
    for name, count in top_pis:
        print(f"{name}: {count}")
    
    # Plot horizontal bar chart
    names, counts = zip(*top_pis)
    fig, ax = plt.subplots()
    ax.barh(names[::-1], counts[::-1])
    ax.set_xlabel('Number of Publications')
    ax.set_title('Top 10 Most Prolific PI')
    plt.tight_layout()
    plt.show()

def top_authors(top_n=10):
    author_series = df['authors'].dropna().str.replace(';', ',', regex=False).str.split(',')
    authors_list = [author.strip() for sublist in author_series for author in sublist if author.strip()]
    for i in authors_list:
        print(i)
    counter = Counter(authors_list)
    top_authors = counter.most_common(top_n)
    
    # Print the top 10 authors
    print("Top 10 authors and publication counts:")
    for name, count in top_authors:
        print(f"{name}: {count}")
    
    # Plot horizontal bar chart
    names, counts = zip(*top_authors)
    fig, ax = plt.subplots()
    ax.barh(names[::-1], counts[::-1])
    ax.set_xlabel('Number of Publications')
    ax.set_title('Top 10 Most Prolific Authors')
    plt.tight_layout()
    plt.show()

def kde_publication_year():

    # Drop missing values in a new DataFrame and convert 'year' to int
    df_clean = df.dropna(subset=['outputyear']).copy()
    df_clean['outputyear'] = df_clean['outputyear'].astype(int)

    # Count publications per year
    pubs_per_year = df_clean['outputyear'].value_counts().sort_index()

    # Create a new DataFrame for KDE
    pubs_df = pd.DataFrame({
        'outputyear': pubs_per_year.index,
        'count': pubs_per_year.values
    })

    # KDE plot using the counts with Gaussian kernel
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=pubs_df,
        x='outputyear',
        weights='count',
        bw_adjust=0.75,
        fill=True
    )
    plt.title("KDE of Publications per Year (Gaussian Kernel)")
    plt.xlabel("Publication Year")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def correlation_map():
    df['author_count'] = (df['authors']
        .fillna('')
        .str.replace(';', ',', regex=False)
        .str.split(',')
        .apply(lambda authors: len([a for a in authors if a.strip()]))
        )
    df['title_len'] = df['outputtitle'].fillna('').str.len()
    df['abstract_len'] = df['abstract'].fillna('').str.len()

    # Prepare correlation matrix
    num_df = df[['outputyear', 'author_count', 'title_len', 'abstract_len']]

    corr = num_df.corr()

    # Plot heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(corr)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax)
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def all_author_graph():
    author_years = {}
    for _, row in df.iterrows():
        try:
            year = int(row['outputyear'])
        except:
            continue
        authors_str = row['authors'] if pd.notnull(row['authors']) else ''
        authors = [a.strip() for a in authors_str.replace(';', ',').split(',') if a.strip()]
        for author in authors:
            author_years.setdefault(author, []).append(year)

    # Build full co-authorship graph
    G_all = nx.Graph()
    for author in author_years:
        G_all.add_node(author)
    for _, row in df.iterrows():
        authors_str = row['authors'] if pd.notnull(row['authors']) else ''
        authors = [a.strip() for a in authors_str.replace(';', ',').split(',') if a.strip()]
        for u, v in combinations(authors, 2):
            if G_all.has_edge(u, v):
                G_all[u][v]['weight'] += 1
            else:
                G_all.add_edge(u, v, weight=1)

    # Compute average publication year per author
    avg_years = np.array([np.mean(author_years.get(node, [np.nan])) for node in G_all.nodes()], dtype=float)

    # Prepare colormap
    vmin, vmax = np.nanmin(avg_years), np.nanmax(avg_years)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis
    default_color = 'lightgray'
    node_colors = [
        cmap(norm(y)) if not np.isnan(y) else default_color
        for y in avg_years
    ]

    # Draw graph
    fig, ax = plt.subplots(figsize=(10,10))
    # pos = nx.spring_layout(G_all, k=0.15, iterations=20, seed=42)
    pos = nx.spring_layout(G_all, k=0.35, iterations=100, seed=42)


    node_sizes = [G_all.degree(n) * 100 for n in G_all.nodes()]
    nx.draw_networkx_nodes(
        G_all, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8
    )
    nx.draw_networkx_edges(
        G_all, pos, ax=ax,
        width=[G_all[u][v]['weight'] * 0.1 for u, v in G_all.edges()],
        alpha=0.3
    )
    nx.draw_networkx_labels(G_all, pos, ax=ax, font_size=4)

    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(avg_years[~np.isnan(avg_years)])
    fig.colorbar(sm, ax=ax, label='Average Publication Year')

    ax.set_title("Co-authorship Network Colored by Avg. Publication Year")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def top100_author_graph(df, top_n=100):
    # 1. Extract all authors into a flat list
    author_series = df['authors'].dropna().str.replace(';', ',', regex=False).str.split(',')
    all_authors = [a.strip() for sublist in author_series for a in sublist if a.strip()]
    
    # 2. Find the top N authors
    author_counts = Counter(all_authors)
    top_authors = {name for name, _ in author_counts.most_common(top_n)}
    
    # 3. Build a co-authorship graph including only those top authors
    G = nx.Graph()
    G.add_nodes_from(top_authors)
    
    for _, row in df.iterrows():
        authors_str = row['authors'] if pd.notnull(row['authors']) else ''
        authors = [a.strip() for a in authors_str.replace(';', ',').split(',') if a.strip()]
        # keep only pairs where both are in top_authors
        authors = [a for a in authors if a in top_authors]
        for u, v in combinations(authors, 2):
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)
    
    # 4. Compute average publication year for color mapping
    author_years = {a: [] for a in top_authors}
    for _, row in df.iterrows():
        try:
            year = int(row['outputyear'])
        except:
            continue
        authors_str = row['authors'] if pd.notnull(row['authors']) else ''
        authors = [a.strip() for a in authors_str.replace(';', ',').split(',') if a.strip()]
        for a in authors:
            if a in author_years:
                author_years[a].append(year)
    avg_years = np.array([
        np.mean(author_years[a]) if author_years[a] else np.nan
        for a in G.nodes()
    ], dtype=float)
    
    # 5. Prepare node colors & sizes
    vmin, vmax = np.nanmin(avg_years), np.nanmax(avg_years)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis
    node_colors = [cmap(norm(y)) if not np.isnan(y) else 'lightgray' for y in avg_years]
    node_sizes  = [G.degree(n) * 100 for n in G.nodes()]
    
    # 6. Draw the graph
    pos = nx.spring_layout(G, k=0.35, iterations=20, seed=42)
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos,
                           width=[d['weight'] * 0.1 for _, _, d in G.edges(data=True)],
                           alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
    
    # Colorbar for average year
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(avg_years[~np.isnan(avg_years)])
    fig.colorbar(sm, ax=ax, label='Average Publication Year')
    
    ax.set_title(f"Top {top_n} Co-authorship Network Colored by Avg. Year")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    df = pd.read_csv('Final_ResearchOutputs_Cleaned.csv')   # or .read_excel, .read_pickle, etc.
    # df = pd.read_csv('enriched_outputs_sample.csv')   # or .read_excel, .read_pickle, etc.

    # print("Columns in DataFrame:")
    # print(df.columns.tolist())

    # print("\nFirst few rows:")
    # print(df.head())

    top_project_rdc()
    publications_by_year()
    top_authors()
    top_pi()
    kde_publication_year()
    correlation_map()
    # all_author_graph()
    top100_author_graph(df, top_n=300)


    
