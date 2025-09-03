"""Dataset preparation functions for the main dataset"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import os
import requests

def load_data() :
    """load the dataframe from kaggle"""
    # Set the path to the file you'd like to load
    file_path = "wikiart_art_pieces.csv"

    # Load the latest version
    df = kagglehub.load_dataset(
      KaggleDatasetAdapter.PANDAS,
      "simolopes/wikiart-all-artpieces",
      file_path
    )

    return df


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """clean the dataframe to keep categories wanted"""

    #lists :
    styles_to_keep = ['Orientalism', 'Neoclassicism', 'Naturalism', 'Romanticism', 'Academicism', 'Neo-Rococo', 'Kitsch', 'Realism', 'Impressionism', 'Expressionism', 'Japonism', 'Magic Realism', 'Symbolism', 'Costumbrismo', 'Biedermeier', 'Luminism', 'Ink and wash painting', 'Naïve Art (Primitivism)', 'Art Nouveau (Modern)', 'Divisionism', 'Social Realism', 'Tonalism', 'Post-Impressionism', 'Pointillism', 'Ukiyo-e', 'American Realism', 'Socialist Realism', 'Fauvism', 'Cubism', 'Abstract Art', 'Precisionism', 'Figurative Expressionism', 'Existential Art', 'Surrealism', 'New Realism', 'Neo-baroque', 'Abstract Expressionism', 'Neo-Romanticism', 'Neo-Impressionism', 'Cloisonnism', 'Synthetism', 'Regionalism', 'Orphism', 'Art Deco', 'Neo-Expressionism', 'Fantasy Art', 'New Medievialism', 'Neo-Byzantine', 'Lyrical Abstraction', 'New Ink Painting', 'Verism', 'Contemporary Realism', 'Color Field Painting', 'Art Brut', 'Outsider art', 'Cubo-Expressionism', 'Pop Art', 'Geometric', 'Art Singulier', 'Action painting', 'Constructivism', 'Neoplasticism', 'Intimism', 'Dada', 'Analytical Cubism', 'Synthetic Cubism', 'Muralism', 'Futurism', 'Cubo-Futurism', 'Tachisme', 'Modernismo', 'Metaphysical art', 'Excessivism', 'Classical Realism', 'Severe Style', 'Miserablism', 'Art Informel', 'Neo-Pop Art', 'Native Art', 'Transavantgarde', 'Contemporary', 'Conceptual Art', 'Light and Space', 'Junk Art', 'Shin-hanga', 'Hard Edge Painting', 'Neo-Figurative Art', 'Purism', 'Tubism', 'Suprematism', 'Concretism', 'Analytical\xa0Realism', 'Mechanistic Cubism', 'Neo-Suprematism', 'Automatic Painting', 'Op Art', 'Minimalism', 'Post-Minimalism', 'Post-Painterly Abstraction', 'Neo-Concretism', 'Lettrism', 'Kinetic Art', 'New European Painting', 'P&D (Pattern and Decoration)', 'New Casualism', 'Neo-Dada', 'Spectralism', 'Rayonism', 'Synchromism', 'Modernism', 'Feminist Art', 'Transautomatism', 'Fantastic Realism', 'Photorealism', 'Hyper-Realism', 'Nouveau Réalisme', 'Postcolonial art', 'Sots Art', 'Indian Space painting', 'Zen', 'Spatialism', 'Cartographic Art', 'Superflat', 'Mail Art', 'Neo-Minimalism', 'Fiber art', 'Street art', 'Neo-Geo', 'Maximalism', 'Queer art', 'Digital Art', 'Cyber Art', 'Poster Art Realism', 'Hyper-Mannerism (Anachronism)', 'Confessional Art', 'Neo-Orthodoxism', 'Graffiti Art', 'Lowbrow Art', 'Stuckism']
    genres_to_drop = ['sketch and study', 'sculpture', 'design', 'installation', 'no genre', 'photo', 'poster', 'caricature', 'graffiti', 'advertisement', 'utensil', 'veduta', 'performance', 'capriccio', 'mural', 'bird-and-flower painting', 'digital', 'architecture', 'mobile', 'miniature', 'tapestry', 'pastorale', 'furniture', 'calligraphy', 'shan shui', 'mosaic', 'vanitas', 'jewelry', 'pin-up', 'video', "trompe-l'œil", 'panorama', 'stabile', 'augmented reality', 'quadratura', 'object', "artist's book", 'ornament', 'animation', 'tronie']

    #Clean the styles
    df_cleaned_styles = df[df['style'].isin(styles_to_keep)]

    #Clean the movement
    df_cleaned_styles_movements = df_cleaned_styles[df_cleaned_styles['movement'].isin(styles_to_keep)]

    #Create a copy of the dataframe and split the genres to a list (multiple values possibles)
    df_cleaned_styles_movements_copy = df_cleaned_styles_movements.copy()
    df_cleaned_styles_movements_copy['genre_list'] = df_cleaned_styles_movements_copy['genre'].str.split(',\s*', regex=True)
    df_cleaned_styles_movements_copy = df_cleaned_styles_movements_copy.drop(columns='genre')

    # Clean the genres (chosed the ones to drop)
    df_cleaned_styles_movement_genres = df_cleaned_styles_movements_copy[df_cleaned_styles_movements_copy['genre_list'].apply(
        lambda x: all(genre not in genres_to_drop for genre in x) if isinstance(x, list) else True
    )]

    df_filtered = df_cleaned_styles_movement_genres.drop_duplicates(subset=['img'], keep='first')
    df_filtered = df_filtered.set_index("file_name", drop=False)
    return df_filtered


def merging_metadata(df: pd.DataFrame) :
    """merge the dataframe with another one with art titles and dates (for the ones that are availables)"""

    # Set the path to the file you'd like to load
    file_path = "wikiart_scraped.csv"

    # Load the latest version
    df2 = kagglehub.load_dataset(
      KaggleDatasetAdapter.PANDAS,
      "antoinegruson/-wikiart-all-images-120k-link",
      file_path
    )

    # df2 cleaning
    df2 = df2.rename(columns={'Link' : 'img'})
    df2 = df2.drop_duplicates(subset=['img'], keep='first')

    # merging
    df_merged = df.merge(df2, how='left', on='img')
    df_merged = df_merged.drop(columns=['Style', 'Artist'])
    df_merged = df_merged.set_index("file_name", drop=False)

    return df_merged

def data_sampling_balanced(df: pd.DataFrame, sample_size, number_styles=10):
    """
    Create a balanced sample in the dataframe:
    - Select the top `number_styles` (by frequency)
    - Sample the same number of rows from each style
    """

    # Identifier les styles les plus fréquents
    top_styles = df['style'].value_counts().head(number_styles).index
    df_topstyles = df[df['style'].isin(top_styles)]

    # Nombre de lignes à tirer par style
    n_per_style = sample_size // number_styles

    # Tirage équilibré
    df_samples = []
    for style in top_styles:
        df_style = df_topstyles[df_topstyles['style'] == style]

        # On prend min(n_per_style, len(df_style)) pour éviter une erreur
        sample_style = df_style.sample(
            n=min(n_per_style, len(df_style)),
            random_state=42,
            replace=False
        )
        df_samples.append(sample_style)

    # Concaténer le tout
    df_balanced = pd.concat(df_samples, axis=0).reset_index(drop=True)
    print(f"created a dataframe of {len(df_balanced)} lines across {number_styles} styles, BALANCED")
    return df_balanced


def data_sampling_balanced_csv(df: pd.DataFrame, sample_size, number_styles=10):
    """
    Create a balanced sample in the dataframe:
    - Select the top `number_styles` (by frequency)
    - Sample the same number of rows from each style
    """

    # Identifier les styles les plus fréquents
    top_styles = df['style'].value_counts().head(number_styles).index
    df_topstyles = df[df['style'].isin(top_styles)]

    # Nombre de lignes à tirer par style
    n_per_style = sample_size // number_styles

    # Tirage équilibré
    df_samples = []
    for style in top_styles:
        df_style = df_topstyles[df_topstyles['style'] == style]

        # On prend min(n_per_style, len(df_style)) pour éviter une erreur
        sample_style = df_style.sample(
            n=min(n_per_style, len(df_style)),
            random_state=42,
            replace=False
        )
        df_samples.append(sample_style)

    # Concaténer le tout
    df_balanced = pd.concat(df_samples, axis=0).reset_index(drop=True)
    file_path=f"data_sampling{sample_size}_topstyles{number_styles}.csv"
    df_balanced.to_csv(file_path, index=False)
    print(f"created a csv of {len(df_balanced)} lines across {number_styles} styles, BALANCED")


def data_sampling(df: pd.DataFrame, sample_size=200, number_styles=10) :
    """create a sample in the dataframe with the top number_styles chosen (bu default 10) and of the chosen sample_size"""

    # Identifier les styles les plus fréquents
    top_styles = df['style'].value_counts().head(number_styles).index
    df_topstyles = df[df['style'].isin(top_styles)]

    # Tirer un échantillon sans doublons
    df_sample_topstyles = df_topstyles.sample(
        n=sample_size,
        random_state=42,
        replace=False
    )

    print(f"!! UNBALANCED DATA. Created a dataframe of {len(df_sample_topstyles)} lines across {number_styles} styles")
    return df_sample_topstyles


def data_sampling_csv(df: pd.DataFrame, sample_size=200, number_styles=10) :
    """create a sample in the dataframe with the top number_styles chosen (bu default 10) and of the chosen sample_size
    Create a CSV"""

    # Identifier les styles les plus fréquents
    top_styles = df['style'].value_counts().head(number_styles).index
    df_topstyles = df[df['style'].isin(top_styles)]

    # Tirer un échantillon sans doublons
    df_sample_topstyles = df_topstyles.sample(
        n=sample_size,
        random_state=42,
        replace=False
    )

    file_path=f"data_sampling{sample_size}_topstyles{number_styles}.csv"
    df_sample_topstyles.to_csv(file_path, index=False)
    print(f"!! UNBALANCED DATA. Created a dataframe of {len(df_sample_topstyles)} lines across {number_styles} styles")



def download_sample_df(df, destination_path) :
    """allow ro download the sample created in a dataframe"""
    # Dossier de destination
    os.makedirs(destination_path, exist_ok=True)

    # Liste pour stocker les échecs
    failed_downloads = []

    for url, file_name in zip(df['img'], df['file_name']):
        file_path = os.path.join(destination_path, file_name)

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded : {file_name}")
        except requests.exceptions.RequestException as e:
            print(f"Impossible to download : {file_name}. Reason : {e}")
            failed_downloads.append(file_name)

    print("Images are downloaded.")
    print(f"\nTotal failed downloads: {len(failed_downloads)}")
    print("Failed files:", failed_downloads)
