import os
from pyspark.sql import functions as F
import numpy as np
from pyspark.sql.types import DoubleType 
from flask import Flask, render_template, jsonify, send_file, request
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode, col, regexp_replace, when, avg , lower , trim
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib
from pyspark.sql.functions import struct
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor as SklearnRF
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

# Set the Python environment path
os.environ['PYSPARK_PYTHON'] = r'C:\Users\wwwkh\.conda\envs\pyspark_env\python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = r'C:\Users\wwwkh\.conda\envs\pyspark_env\python.exe'
# Set the style for all visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.master('local').appName('House Price').getOrCreate()

def load_data():
    """Load and clean the data."""
    data = spark.read.csv('houcingData.csv', inferSchema=True, header=True)
    
    df = data.withColumn("Surface totale", regexp_replace(col("Surface totale"), " m²", "").cast("int"))
    df = df.withColumn("prix", regexp_replace(col("prix"), "[^0-9.]", "").cast("float"))
    df = df.withColumn("Chambres", col("Chambres").cast("float"))
    df = df.withColumn("Salle de bain", col("Salle de bain").cast("float"))
    df = df.withColumn("Subcategories",when(col("Subcategories") == "villa & riad", "villa&riad").otherwise(col("Subcategories")))
    df = df.na.drop()
    categorical_columns = ['ville','secteur', 'categories', 'Subcategories']
    # Create StringIndexer with handleInvalid set to "skip"
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") for col in categorical_columns]
    # Création d'un pipeline pour appliquer les transformations
    pipeline = Pipeline(stages=indexers)
    # Application du pipeline
    indexed_model = pipeline.fit(df)
    df = indexed_model.transform(df)
    return df

def save_plot(fig, plot_type):
    """Helper function to save plot and return file path."""
    temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f'{plot_type}_{os.getpid()}.png')
    
    if os.path.exists(file_path):
        os.remove(file_path)
    
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return file_path

def create_plot_template(figsize=(12, 7)):
    """Create a template for plots with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig, ax

@app.route('/')
def home():
    # Render home.html as the main page
    return render_template('home.html')
@app.route('/home2')
def home2():
    # Render home2.html
    return render_template('home2.html')

@app.route('/index')
def index():
    # Render index.html
    return render_template('index.html')
@app.route('/recommendation')
def recommendation():
    # Render index.html
    return render_template('recommendation.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/analysis')
def analysis():
    # Render analysis.html
    return render_template('analysis.html')
   
@app.route('/api/properties')
def get_properties():
    df = load_data()
    properties=df.select("nom",
        "prix",
        "ville",
        "secteur",
        "date",
        "Chambres",
        "Salle de bain",
        "Surface totale",
        "link",
        "categories",
        "Subcategories")
    return jsonify(properties.toPandas().to_dict(orient='records'))

@app.route('/plot/price_by_subcategories')
def price_by_subcategories():
    df = load_data()
    df_avg_price_by_type = df.groupBy("Subcategories").agg(avg("prix").alias("avg_price"))
    df_avg_price_by_type_pandas = df_avg_price_by_type.toPandas()

    fig, ax = create_plot_template(figsize=(14, 8))
    sns.barplot(x='Subcategories', y='avg_price', data=df_avg_price_by_type_pandas, ax=ax)
    
    plt.title("Prix Moyen par Subcategories", pad=20, fontweight='bold')
    plt.xlabel("Subcategories", labelpad=15)
    plt.ylabel("Prix Moyen (MAD)", labelpad=15)
    plt.xticks(rotation=45, ha='right')
    
    for i, v in enumerate(df_avg_price_by_type_pandas['avg_price']):
        ax.text(i, v, f'{v:,.2f}MAD', ha='center', va='bottom', fontsize=10)
    
    fig.tight_layout(pad=2)
    file_path = save_plot(fig, 'subcategories')
    return send_file(file_path, mimetype='image/png')
@app.route('/api/cities')
def get_cities():
    df = load_data()
    # Extraire la liste unique des villes
    cities = df.select("ville").distinct().rdd.flatMap(lambda x: x).collect()
    return jsonify(cities)

@app.route('/plot/price_by_categories')
def price_by_categories():
    df = load_data()
    df_avg_price_by_categories = df.groupBy('categories').agg(avg('prix').alias('avg_price'))
    df_avg_price_by_categories_pandas = df_avg_price_by_categories.toPandas()

    fig, ax = create_plot_template(figsize=(14, 8))
    sns.barplot(x='categories', y='avg_price', data=df_avg_price_by_categories_pandas, ax=ax)
    
    plt.title("Prix Moyen par Catégorie", pad=20, fontweight='bold')
    plt.xlabel("Catégorie", labelpad=15)
    plt.ylabel("Prix Moyen (MAD)", labelpad=15)
    plt.xticks(rotation=45, ha='right')
    
    for i, v in enumerate(df_avg_price_by_categories_pandas['avg_price']):
        ax.text(i, v, f'{v:,.2f}MAD', ha='center', va='bottom', fontsize=10)
    
    fig.tight_layout(pad=2)
    file_path = save_plot(fig, 'categories')
    return send_file(file_path, mimetype='image/png')

@app.route('/plot/price_by_ville')
def price_by_ville():
    df = load_data()
    df_avg_price_by_ville = df.groupBy('ville').agg(avg('prix').alias('avg_price'))
    df_avg_price_by_ville_pandas = df_avg_price_by_ville.toPandas()

    fig, ax = plt.subplots(figsize=(16, 8))  # Augmenter la taille pour plus de lisibilité
    sns.barplot(x='ville', y='avg_price', data=df_avg_price_by_ville_pandas, palette='viridis', ax=ax)

    # Rotation des noms des villes
    plt.xticks(rotation=90, ha='center')  # Rotation de 90 degrés, centrée

    # Ajout des titres et labels
    plt.title("Prix Moyen par Ville", pad=20, fontweight='bold')
    plt.xlabel("Ville", labelpad=15)
    plt.ylabel("Prix Moyen (MAD)", labelpad=15)

    plt.tight_layout()

    # Sauvegarder le graphique
    file_path = save_plot(fig, 'price_by_ville')
    return send_file(file_path, mimetype='image/png')
@app.route('/plot/louer_vs_vente/<type>') 
def louer_vs_vente(type):
    # Charger les données
    df = load_data()

    # Filtrer les données pour obtenir seulement les maisons
    df_maison = df.filter(df["Subcategories"] == type)

    # Compter le nombre de maisons à vendre et à louer
    maison_louer = df_maison.filter(df_maison["categories"] == "louer").count()
    maison_vente = df_maison.filter(df_maison["categories"] == "vendre").count()

    # Calculer les pourcentages
    total = maison_louer + maison_vente
    pourcentage_louer = (maison_louer / total) * 100
    pourcentage_vente = (maison_vente / total) * 100

    # Créer le graphique en camembert
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = [f'{type} à Louer', f'{type} à Vendre']
    sizes = [pourcentage_louer, pourcentage_vente]
    colors = ['#FF9999', '#66B3FF']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    ax.axis('equal')  # Assure que le graphique est circulaire

    # Titre du graphique
    

    # Sauvegarder le graphique
    file_path = save_plot(fig, f'{type}_louer_vs_vente')

    # Retourner l'image générée
    return send_file(file_path, mimetype='image/png')
@app.route('/plot/avg_rooms_by_subcategory/<ville>')
def avg_rooms_by_subcategory(ville):

    # Charger les données
    df = load_data()

    # Filtrer les données pour obtenir les maisons de la ville spécifiée
    df_ville = df.filter(df["ville"] == ville)

    # Calculer la moyenne du nombre de chambres par sous-catégorie
    avg_rooms_by_subcategory = df_ville.groupBy("Subcategories").agg({"Chambres": "avg"})

    # Convertir le résultat en un format exploitable pour le graphique
    avg_rooms_by_subcategory_pd = avg_rooms_by_subcategory.toPandas()

    # Créer un diagramme à barres
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(avg_rooms_by_subcategory_pd["Subcategories"], avg_rooms_by_subcategory_pd["avg(Chambres)"], color='skyblue')

    # Ajouter des titres et labels
    ax.set_title(f'Moyenne du nombre de chambres par sous-catégorie pour la ville de {ville}')
    ax.set_xlabel('Sous-catégorie')
    ax.set_ylabel('Moyenne du nombre de chambres')

    # Rotation des labels sur l'axe X pour les rendre plus lisibles
    plt.xticks(rotation=45, ha='right')

    # Sauvegarder le graphique
    file_path = save_plot(fig, f'avg_rooms_by_subcategory_{ville}')

    # Retourner l'image générée
    return send_file(file_path, mimetype='image/png')
@app.route('/plot/avg_surface_by_subcategory/<ville>')
def avg_surface_by_subcategory(ville):

    # Charger les données
    df = load_data()

    # Filtrer les données pour obtenir les maisons de la ville spécifiée
    df_ville = df.filter(df["ville"] == ville)

    # Calculer la moyenne du nombre de chambres par sous-catégorie
    avg_rooms_by_subcategory = df_ville.groupBy("Subcategories").agg({"Surface totale": "avg"})

    # Convertir le résultat en un format exploitable pour le graphique
    avg_rooms_by_subcategory_pd = avg_rooms_by_subcategory.toPandas()

    # Créer un diagramme à barres
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(avg_rooms_by_subcategory_pd["Subcategories"], avg_rooms_by_subcategory_pd["avg(Surface totale)"], color='skyblue')

    # Ajouter des titres et labels
    ax.set_title(f'Moyenne de la surface par sous-catégorie pour la ville de {ville}')
    ax.set_xlabel('Sous-catégorie')
    ax.set_ylabel('Surface')

    # Rotation des labels sur l'axe X pour les rendre plus lisibles
    plt.xticks(rotation=45, ha='right')

    # Sauvegarder le graphique
    file_path = save_plot(fig, f'avg_surface_by_subcategory_{ville}')

    # Retourner l'image générée
    return send_file(file_path, mimetype='image/png')

@app.route('/plot/price_by_rooms')
def price_by_rooms():
    df = load_data()
    df_pandas = df.select("prix", "Chambres").toPandas()

    fig, ax = create_plot_template(figsize=(14, 8))
    sns.scatterplot(data=df_pandas, x="Chambres", y="prix", alpha=0.6, s=100)
    
    plt.title("Relation entre Prix et Nombre de Chambres", pad=20, fontweight='bold')
    plt.xlabel("Nombre de Chambres", labelpad=15)
    plt.ylabel("Prix (MAD)", labelpad=15)
    
    sns.regplot(data=df_pandas, x="Chambres", y="prix", scatter=False, color='red', line_kws={'linestyle': '--'})
    fig.tight_layout(pad=2)
    
    file_path = save_plot(fig, 'rooms')
    return send_file(file_path, mimetype='image/png')

@app.route('/plot/price_by_bathrooms')
def price_by_bathrooms():
    df = load_data()
    df_pandas = df.select("prix", "Salle de bain").toPandas()

    fig, ax = create_plot_template(figsize=(14, 8))
    sns.scatterplot(data=df_pandas, x="Salle de bain", y="prix", alpha=0.6, s=100)
    
    plt.title("Relation entre Prix et Nombre de Salles de Bain", pad=20, fontweight='bold')
    plt.xlabel("Nombre de Salles de Bain", labelpad=15)
    plt.ylabel("Prix (MAD)", labelpad=15)
    
    sns.regplot(data=df_pandas, x="Salle de bain", y="prix", scatter=False, color='red', line_kws={'linestyle': '--'})
    fig.tight_layout(pad=2)
    
    file_path = save_plot(fig, 'bathrooms')
    return send_file(file_path, mimetype='image/png')

@app.route('/plot/price_by_surface')
def price_by_surface():
    df = load_data()
    df_pandas = df.select("prix", "Surface totale").toPandas()

    fig, ax = create_plot_template(figsize=(14, 8))
    sns.scatterplot(data=df_pandas, x="Surface totale", y="prix", alpha=0.6, s=100)
    
    plt.title("Relation entre Prix et Surface Totale", pad=20, fontweight='bold')
    plt.xlabel("Surface Totale (m²)", labelpad=15)
    plt.ylabel("Prix (MAD)", labelpad=15)
    
    sns.regplot(data=df_pandas, x="Surface totale", y="prix", scatter=False, color='red', line_kws={'linestyle': '--'})
    fig.tight_layout(pad=2)
    
    file_path = save_plot(fig, 'surface')
    return send_file(file_path, mimetype='image/png')
@app.route('/plot/recommendation_houcing/<ville>/<budget>/<chambre>')
def recommendation_houcing(ville, budget, chambre):
    df = load_data()
    budget = float(budget)  # Convertir budget en float
    chambre = int(chambre)
    
    # Trouver l'index de la ville
    ville_indices = df.filter(df["ville"] == ville).select("ville_index").distinct().collect()
    if len(ville_indices) != 1:
        raise ValueError(f"Ambiguity in ville_index for {ville}: {ville_indices}")
    v_index = ville_indices[0][0]
    print(ville, "=", v_index)
    
    # Calculer la note
    df_rating = df.withColumn("rating", col("prix") / 1000000)
    
    # Diviser les données en ensemble d'entraînement et de test
    (training, test) = df_rating.randomSplit([0.8, 0.2])
    
    # Configurer ALS
    als = ALS(
        maxIter=10,
        regParam=0.1,
        userCol="ville_index",
        itemCol="Subcategories_index",
        ratingCol="rating",
        coldStartStrategy="drop"
    )
    
    # Entraîner le modèle
    model = als.fit(training)
    
    # Récupérer les recommandations
    user_recommendations = model.recommendForAllUsers(5)
    item_recommendations = model.recommendForAllItems(5)
    
    # Applatir les recommandations
    user_recommendations_flat = user_recommendations.withColumn("recommendation", explode(col("recommendations")))
    user_recommendations_flat = user_recommendations_flat.select(
        "ville_index",
        col("recommendation.Subcategories_index").alias("Subcategories_index"),
        col("recommendation.rating").alias("rating")
    )
    
    # Filtrer les recommandations par ville_index (ville donnée)
    filtered_recommendations = user_recommendations_flat.filter(col("ville_index") == v_index)
    
    # Joindre avec df_rating pour obtenir les détails de la maison
    final_recommendations = filtered_recommendations.join(df_rating, on="Subcategories_index")
    
    # Sélectionner uniquement les colonnes souhaitées
    final_recommendations = final_recommendations.select(
        "nom",
        "prix",
        "ville",
        "secteur",
        "date",
        "Chambres",
        "Salle de bain",
        "Surface totale",
        "link",
        "categories",
        "Subcategories"
    )
    
    # Filtrer par budget et nombre de chambres
    final_recommendations = final_recommendations.filter(col("ville") == ville)
    final_recommendations = final_recommendations.filter(col("prix") <= budget)
    final_recommendations = final_recommendations.filter(col("Chambres") >= chambre)
    
    # Retourner les résultats sous forme de JSON
    return jsonify(final_recommendations.toPandas().to_dict(orient='records'))
@app.route('/plot/price_prediction/<ville>/<property>/<area>')
def price_prediction(ville, property, area):
    """
    Predicts the price of a property based on city, property type, and area.
    """
    try:
        df = load_data()
        area = float(area)  # Ensure area is a float

        # Filter the data for the given city and property type
        filtered_df = df.filter((col("ville") == ville) & (col("Subcategories") == property))

        if filtered_df.count() == 0:
            return jsonify({"error": f"No data available for city '{ville}' and property type '{property}'"}), 404

        # Define features and label for prediction
        vector_assembler = VectorAssembler(
            inputCols=["Surface totale", "ville_index", "Subcategories_index"],
            outputCol="features"
        )
        model_data = vector_assembler.transform(filtered_df)

        # Define Random Forest Regressor
        rf = RandomForestRegressor(
            featuresCol="features", labelCol="prix", predictionCol="prediction"
        )

        # Train-Test split
        train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=123)

        # Fit the model
        rf_model = rf.fit(train_data)

        # Prepare input features for prediction
        property_index = filtered_df.select("Subcategories_index").first()[0]
        city_index = filtered_df.select("ville_index").first()[0]

        prediction_data = spark.createDataFrame(
            [(area, city_index, property_index)],
            ["Surface totale", "ville_index", "Subcategories_index"]
        )

        prediction_features = vector_assembler.transform(prediction_data)

        # Predict the price
        prediction = rf_model.transform(prediction_features).select("prediction").first()[0]

        # Generate a plot of similar property prices
        similar_properties = filtered_df.filter((col("Surface totale") >= area - 20) & (col("Surface totale") <= area + 20)).toPandas()

        fig, ax = create_plot_template()
        sns.histplot(similar_properties["prix"], bins=15, kde=True, ax=ax, color="skyblue")

        plt.axvline(prediction, color="red", linestyle="--", label=f"Predicted Price: {prediction:,.2f} MAD")
        plt.title(f"Predicted Price for {property} in {ville}", pad=20, fontweight="bold")
        plt.xlabel("Price (MAD)", labelpad=15)
        plt.ylabel("Frequency", labelpad=15)
        plt.legend()

        file_path = save_plot(fig, f"price_prediction_{ville}_{property}_{area}")
        return send_file(file_path, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/properties_by_city/<city>')
def get_properties_by_city(city):
    """Get available property types for a specific city"""
    df = load_data()
    
    # Get distinct property types for the specified city
    property_types = df.filter(lower(trim(col("ville"))) == city.lower()) \
                      .select("Subcategories") \
                      .distinct() \
                      .rdd.flatMap(lambda x: x).collect()
    
    return jsonify(property_types)
@app.route('/plot/enhanced_prediction/<ville>/<property>/<area>/<bedrooms>/<bathrooms>')
def enhanced_prediction(ville, property, area, bedrooms, bathrooms):
    """
    Enhanced price prediction using clustering and multiple features
    """
    try:
        df = load_data()
        
        # Convert inputs to appropriate types
        area = float(area)
        bedrooms = float(bedrooms)
        bathrooms = float(bathrooms)

        # Clean and standardize property types
        df = df.withColumn("Subcategories", 
            when(col("Subcategories") == "appartements", "appartements")
            .when(col("Subcategories") == "villa & riad", "villa&riad")
            .otherwise(col("Subcategories")))

        # Filter out rows with null or empty prices
        df = df.filter(
            (col("prix").isNotNull()) & 
            (col("prix") != "") & 
            (col("prix").cast("float").isNotNull())
        )

        # Filter data for the given city and property type
        filtered_df = df.filter(
            (lower(col("ville")).contains(ville.lower())) & 
            (lower(col("Subcategories")) == property.lower())
        )

        if filtered_df.count() == 0:
            return jsonify({
                "error": f"No data available for {property} in {ville}. " +
                        f"Available property types: {', '.join(df.select('Subcategories').distinct().rdd.flatMap(lambda x: x).collect())}"
            }), 404

        # Create feature vector for clustering
        vector_assembler = VectorAssembler(
            inputCols=["Surface totale", "Chambres", "Salle de bain", "ville_index", "Subcategories_index"],
            outputCol="features"
        )
        
        # Handle null values in numerical columns
        df_cleaned = filtered_df.na.fill({
            "Surface totale": filtered_df.select(avg("Surface totale")).first()[0],
            "Chambres": filtered_df.select(avg("Chambres")).first()[0],
            "Salle de bain": filtered_df.select(avg("Salle de bain")).first()[0]
        })

        # Standardize features
        from pyspark.ml.feature import StandardScaler
        standardizer = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )

        # Add K-means clustering
        from pyspark.ml.clustering import KMeans
        # Adjust number of clusters based on data size
        k = min(5, filtered_df.count() // 5) if filtered_df.count() > 5 else 2
        kmeans = KMeans(k=k, featuresCol="scaled_features", predictionCol="cluster")

        # Create pipeline
        from pyspark.ml import Pipeline
        pipeline = Pipeline(stages=[
            vector_assembler,
            standardizer,
            kmeans
        ])

        # Fit pipeline
        pipeline_model = pipeline.fit(df_cleaned)
        clustered_df = pipeline_model.transform(df_cleaned)

        # Train separate RF models for each cluster
        rf_models = {}
        evaluators = {}
        
        for cluster_id in range(k):
            cluster_data = clustered_df.filter(col("cluster") == cluster_id)
            if cluster_data.count() > 0:
                # Prepare features for RF
                rf_assembler = VectorAssembler(
                    inputCols=["Surface totale", "Chambres", "Salle de bain", "ville_index", "Subcategories_index"],
                    outputCol="rf_features"
                )
                rf_data = rf_assembler.transform(cluster_data)

                # Split data
                train_data, test_data = rf_data.randomSplit([0.8, 0.2], seed=123)

                # Train RF model
                rf = RandomForestRegressor(
                    featuresCol="rf_features",
                    labelCol="prix",
                    numTrees=50,
                    maxDepth=5
                )
                rf_models[cluster_id] = rf.fit(train_data)

                # Evaluate model
                predictions = rf_models[cluster_id].transform(test_data)
                evaluator = RegressionEvaluator(
                    labelCol="prix",
                    predictionCol="prediction",
                    metricName="rmse"
                )
                evaluators[cluster_id] = evaluator.evaluate(predictions)

        # Prepare input data for prediction
        input_data = spark.createDataFrame(
            [(area, bedrooms, bathrooms, 
              filtered_df.select("ville_index").first()[0],
              filtered_df.select("Subcategories_index").first()[0])],
            ["Surface totale", "Chambres", "Salle de bain", "ville_index", "Subcategories_index"]
        )

        # Transform input data
        transformed_input = pipeline_model.transform(input_data)
        cluster_id = transformed_input.select("cluster").first()[0]

        # Prepare features for prediction
        rf_assembler = VectorAssembler(
            inputCols=["Surface totale", "Chambres", "Salle de bain", "ville_index", "Subcategories_index"],
            outputCol="rf_features"
        )
        prediction_features = rf_assembler.transform(transformed_input)

        # Get prediction
        if cluster_id in rf_models:
            prediction = rf_models[cluster_id].transform(prediction_features).select("prediction").first()[0]
            rmse = evaluators[cluster_id]
        else:
            # Fallback to nearest cluster if current cluster has no model
            cluster_id = min(rf_models.keys())
            prediction = rf_models[cluster_id].transform(prediction_features).select("prediction").first()[0]
            rmse = evaluators[cluster_id]

        # Get similar properties in the same cluster
        similar_properties = clustered_df.filter(
            (col("cluster") == cluster_id) &
            (col("Surface totale").between(area - 20, area + 20)) &
            (col("Chambres").between(bedrooms - 1, bedrooms + 1)) &
            (col("Salle de bain").between(bathrooms - 1, bathrooms + 1))
        ).toPandas()

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Price distribution plot
        sns.histplot(similar_properties["prix"], bins=15, kde=True, ax=ax1, color="skyblue")
        ax1.axvline(prediction, color="red", linestyle="--", 
                   label=f"Predicted: {prediction:,.2f} MAD\nRMSE: {rmse:,.2f}")
        ax1.set_title("Price Distribution in Cluster")
        ax1.set_xlabel("Price (MAD)")
        ax1.set_ylabel("Frequency")
        ax1.legend()

        # Scatter plot of area vs price
        sns.scatterplot(data=similar_properties, x="Surface totale", y="prix", ax=ax2, alpha=0.6)
        ax2.scatter(area, prediction, color="red", marker="*", s=200, 
                   label="Predicted Property")
        ax2.set_title("Area vs Price in Cluster")
        ax2.set_xlabel("Surface Area (m²)")
        ax2.set_ylabel("Price (MAD)")
        ax2.legend()

        plt.tight_layout()
        file_path = save_plot(fig, f"enhanced_prediction_{ville}_{property}")
        
        return jsonify({
            "prediction": prediction,
            "rmse": rmse,
            "cluster_id": int(cluster_id),
            "similar_properties": similar_properties.to_dict(orient="records"),
            "plot": file_path
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # This will print the full error trace
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
