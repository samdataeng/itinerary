## local files needed to run the script:
## ontology.xml
## a .csv file, e.g. datatourisme-reg-reu-main.csv

import xml.etree.ElementTree as ET
import json
from pathlib import Path
import json
import sys
import re
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from fastapi import FastAPI
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Optional
from pydantic import BaseModel, Field, Extra
from fastapi.param_functions import Query
from collections import defaultdict

api = FastAPI(
    title="Ton séjour à La Réunion ",
    description="Proposition de programmes pour vos séjours à La Réunion ",
)

security = HTTPBasic()


@api.get("/etat")
def get_state():
    """
    Checks API state
    """
    return "API is working"


responses = {200: {"description": "OK"}, 400: {"description": "Bad request"}}


def validate_admin_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username == "admin" and credentials.password == "itineraire2023":
        return "Admin Successfully logged in"
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Basic"},
    )


@api.post("/database_update", dependencies=[Depends(validate_admin_credentials)])
def database_update(
    file: str = Query(
        "datatourisme-reg-reu-20230312.csv",
        title=".csv file from datatourisme",
        description=".csv file with POIs, filename must be datatourisme-reg-reu-YYYYMMDD.csv",
    )
):
    # print("api", api.state.content)
    """
    This endpoint can be used by admin only to update the database
    - INPUT file has to be: .csv (datatourisme-reg-reu-YYYYMMDD.csv, for example datatourisme-reg-reu-20230312.csv) dowloaded from
    https://www.data.gouv.fr/fr/datasets/datatourisme-la-base-nationale-des-donnees-publiques-dinformation-touristique-en-open-data/
    """
    if not os.path.exists(file):
        raise HTTPException(
            status_code=400, detail="File not found. Default file will be used."
        )

    elif not re.match(r"datatourisme-reg-reu-\d{8}\.csv", file):
        raise HTTPException(
            status_code=400, detail="Invalid input file. Default file will be used."
        )

    else:
        shutil.copy(file, "datatourisme-reg-reu-main.csv")
        return f"Database succesfully updated"


def compute_df_cat_en_fr(root):
    # extraction of English and French labels for categories
    categories = []
    for e in root.findall("{http://www.w3.org/2002/07/owl#}Class"):
        url = e.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")
        category_code = url.split("#")[-1]
        category_fr = ""
        category_en = ""
        for f in e.findall("{http://www.w3.org/2000/01/rdf-schema#}label"):
            if f.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") == "fr":
                category_fr = f.text
            if f.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") == "en":
                category_en = f.text
        if category_fr or category_en:
            categories.append((category_code, category_en, category_fr))
    df_cat_en_fr = pd.DataFrame(
        categories, columns=["category_code", "category_en", "category_fr"]
    )
    return df_cat_en_fr


def get_classes_to_be_removed_list(root):
    master_categories = {}
    for e in root.findall("{http://www.w3.org/2002/07/owl#}Class"):
        url = e.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")
        if "datatourisme" not in url:
            continue
        sub_category_code = url.split("#")[-1]
        for sub_class_of in e.findall(
            "{http://www.w3.org/2000/01/rdf-schema#}subClassOf"
        ):
            sub_class_of_url = sub_class_of.get(
                "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
            )
            if sub_class_of_url and "datatourisme" in sub_class_of_url:
                master_category_code = sub_class_of_url.split("#")[-1]
                master_categories[sub_category_code] = master_category_code

    # generation of complete hierarchical lists
    no_place_of_interest = []
    all_reversed_categories_chains = []
    num_line = 0
    full_categories_chain_list = []
    with open("ontology_hierarchy_descending.csv", "w") as file1:
        for sub_category, master_category in master_categories.items():
            if (
                sub_category not in master_categories.values()
            ):  # test to print only final categories
                full_categories_chain = [sub_category, master_category]
                while (
                    master_category in master_categories
                ):  # iterate as long as there is a master category
                    master_category = master_categories[master_category]
                    full_categories_chain.append(master_category)
                if "PlaceOfInterest" in full_categories_chain and not any(
                    category in full_categories_chain
                    for category in [
                        "Accommodation",
                        "BusinessPlace",
                        "ConvenientService",
                        "MedicalPlace",
                        "ServiceProvider",
                        "TastingProvider",
                        "Transport",
                        "TouristInformationCenter",
                    ]
                ):
                    full_categories_chain_list.append(full_categories_chain)
                    reversed_categories_chain = full_categories_chain[::-1]
                    all_reversed_categories_chains.append(
                        reversed_categories_chain
                    )  # keep reversed lists to print later
                else:
                    if "PlaceOfInterest" in full_categories_chain:
                        full_categories_chain.remove("PlaceOfInterest")
                    if "PointOfInterest" in full_categories_chain:
                        full_categories_chain.remove("PointOfInterest")
                    no_place_of_interest = (
                        no_place_of_interest + full_categories_chain
                    )  # terms (subclasses of accommodation) to be removed from all lists
        all_reversed_categories_chains_sorted = sorted(all_reversed_categories_chains)
        for i in range(len(all_reversed_categories_chains_sorted)):
            print(
                ",".join(all_reversed_categories_chains_sorted[i]), file=file1
            )  ## print reversed ontology (starting from general category), alphabetically sorted, as csv

    classes_to_be_removed = set(no_place_of_interest)  # unique
    classes_to_be_removed_list = list(classes_to_be_removed)  # list
    return (
        master_categories,
        classes_to_be_removed_list,
        full_categories_chain_list,
        all_reversed_categories_chains_sorted,
    )


def write_files(master_categories, classes_to_be_removed_list):
    allclasses = []

    with open("subclass_class_relations.csv", "w") as file2:
        print("subclass,class", file=file2)
        for sub_category, master_category in master_categories.items():
            check = any(
                item in [sub_category, master_category]
                for item in classes_to_be_removed_list
            )
            if check:
                pass  # don't print (unwanted category)
            else:
                print(
                    ",".join([sub_category, master_category]), file=file2
                )  # print relations between categories
                allclasses = allclasses + [sub_category, master_category]

    with open("allclasses.csv", "w") as file3:
        print("class", file=file3)
        print(
            "\n".join(sorted(list(set(allclasses)))), file=file3
        )  # print unique categories


# extract the categories of the POI from the URL
def splitter(row):
    l = row.split("|")
    m = [i.split("#") for i in l]
    cats = [j[1] for j in m if len(j) == 2]
    return cats


def filter_and_keep_lowest_cat(
    all_categories, excluded_categories, categories, is_higher_than
):
    lower_categories = []
    for i in range(0, len(categories)):
        is_lower_category = True
        if categories[i] not in all_categories or categories[i] in excluded_categories:
            continue
        for j in range(0, len(categories)):
            if i != j and categories[j] in all_categories:
                if categories[j] in is_higher_than.get(categories[i], []):
                    is_lower_category = False
        if is_lower_category:
            lower_categories.append(categories[i])
    return lower_categories


def add_global_cat(cat_chain_list, categories):
    lowest_cats = categories
    highest_cats = []
    for cat in lowest_cats:
        for l in cat_chain_list:
            if cat in l:
                highest_cats.append(l[2])
                break
    return highest_cats


def prepare_dataframes(
    df_cat_en_fr,
    fixedFile,
    full_categories_chain_list,
    all_reversed_categories_chains_sorted,
):
    # create a dataframe from the csv
    df = pd.read_csv(fixedFile, sep=",", header=0)
    # create a column with the number of the region
    df["region"] = df["URI_ID_du_POI"].apply(lambda x: x.rsplit("/", 2)[1])
    # create a column with the ID (to be used as primary key)
    df["ID"] = df["URI_ID_du_POI"].apply(lambda x: x.rsplit("/", 2)[2])
    # split the column code_postal_et_commune into 2 columns: code_postal / commune
    df["Code_postal_et_commune"] = df["Code_postal_et_commune"].apply(
        lambda x: x.split("#")
    )
    df["commune"] = df["Code_postal_et_commune"].apply(lambda x: x[1])
    df["code_postal"] = df["Code_postal_et_commune"].apply(lambda x: x[0])
    df["categories"] = df["Categories_de_POI"].apply(lambda row: splitter(row))

    # filter categories to keep the finest one
    cat_chain_list = full_categories_chain_list
    is_higher_than = defaultdict(set)
    all_categories = set()
    for l in cat_chain_list:
        if "PlaceOfInterest" in l:
            l.remove("PlaceOfInterest")
        if "PointOfInterest" in l:
            l.remove("PointOfInterest")
        for i in range(0, len(l)):
            all_categories.add(l[i])
            for j in range(i + 1, len(l)):
                is_higher_than[l[j]].add(l[i])

    excluded_categories = ["FitnessCenter"]

    df["categories"] = df["categories"].apply(
        lambda categories: filter_and_keep_lowest_cat(
            all_categories, excluded_categories, categories, is_higher_than
        )
    )

    # add the most general category (apart from PlaceOfInterest and PointOfInterest) in a new column
    cat_chain_list = all_reversed_categories_chains_sorted

    df["supercategories"] = df["categories"].apply(
        lambda categories: add_global_cat(cat_chain_list, categories)
    )

    # check the maximum number of categories
    max_cats = max(len(i) for i in df["categories"])

    # split the list of categories into several columns
    categories = pd.DataFrame(
        df["categories"].to_list(),
        columns=["subclass_" + str(i) for i in range(max_cats)],
    )

    # split the list of supercategories into several columns
    supercategories = pd.DataFrame(
        df["supercategories"].to_list(),
        columns=["superclass_" + str(i) for i in range(max_cats)],
    )

    # add the columns categories and supercategories to the df
    df = pd.concat([df, categories, supercategories], axis=1)

    # delete lines without category
    df = df[~(df["categories"].apply(len) == 0)]

    # delete irrelevant and empty columns
    df = df.drop(
        columns=[
            "categories",
            "supercategories",
            "Code_postal_et_commune",
            "Covid19_mesures_specifiques",
            "Covid19_est_en_activite",
            "Covid19_periodes_d_ouvertures_confirmees",
            "Categories_de_POI",
            "Createur_de_la_donnee",
            "SIT_diffuseur",
            "Classements_du_POI",
            "URI_ID_du_POI",
        ]
    )
    df = df.dropna(axis=1, how="all")

    # delete duplicates
    df = df.drop_duplicates(subset=['Nom_du_POI', 'Latitude', 'Longitude'])

    # add columns with FR labels for categories
    merged_df = pd.merge(
        df,
        df_cat_en_fr[["category_code", "category_fr"]],
        left_on="subclass_0",
        right_on="category_code",
    ).drop("category_code", axis=1)
    merged_df = merged_df.rename(columns={"category_fr": "subclass_0_fr"})
    merged_df = pd.merge(
        merged_df,
        df_cat_en_fr[["category_code", "category_fr"]],
        left_on="superclass_0",
        right_on="category_code",
    ).drop("category_code", axis=1)
    merged_df = merged_df.rename(columns={"category_fr": "superclass_0_fr"})

    # define the ID as index of the df
    ##df_clean=df.set_index('ID',drop=True)
    df_clean = merged_df.set_index("ID", drop=True)

    # create a "Food" dataframe
    df_food = df_clean[
        (df_clean["superclass_0"] == "FoodEstablishment")
        | (df_clean["superclass_1"] == "FoodEstablishment")
    ]

    # create a dataframe with POI only (exclude food)
    df_poi = df_clean[
        (~df_clean[["superclass_0", "superclass_1"]].isin(["FoodEstablishment"])).all(1)
    ]

    # create an "Air activity" dataframe (paragliding, ultralight aviation etc)
    df_airactivity = df_poi[
        (
            (df_poi["subclass_0"] == "LeisureSportActivityProvider")
            | (df_poi["subclass_1"] == "LeisureSportActivityProvider")
        )
        & (
            (df_poi["Nom_du_POI"].str.contains("Parapente"))
            | (df_poi["Nom_du_POI"].str.contains("Air"))
            | (df_poi["Nom_du_POI"].str.contains("Aéro"))
            | (df_poi["Nom_du_POI"].str.contains("Paramoteur"))
            | (df_poi["Nom_du_POI"].str.contains("Ulm"))
            | (df_poi["Nom_du_POI"].str.contains("ULM"))
            | (df_poi["Nom_du_POI"].str.contains("Héli"))
            | (df_poi["Nom_du_POI"].str.contains("Aile"))
            | (df_poi["Adresse_postale"].str.contains("ULM"))
        )
    ]

    # create an "Outdoor" dataframe (air activity, nautical centre, natural heritage (subclass))
    # keep waterfall, mountain, beach etc as POI; classify NaturalHeritage without subclass as activity
    df_nauticalcentre = df_poi[
        (df_poi["subclass_0"] == "NauticalCentre")
        | (df_poi["subclass_1"] == "NauticalCentre")
    ]
    df_subnaturalheritage = df_poi[
        (df_poi["subclass_0"] == "NaturalHeritage")
        | (df_poi["subclass_1"] == "NaturalHeritage")
    ]
    df_outdooractivity = df_airactivity.append(
        [df_nauticalcentre, df_subnaturalheritage]
    )

    # create a df with POI only (exclude food, store and outdoor)
    df_poi_2 = pd.concat([df_poi, df_outdooractivity]).drop_duplicates(keep=False)
    df_poi_2 = df_poi_2[
        (
            ~df_poi_2[["superclass_0", "superclass_1"]].isin(
                [
                    "ActivityProvider",
                    "FoodEstablishment",
                    "SportsAndLeisurePlace",
                    "Store",
                ]
            )
        ).all(1)
    ]

    # create a df with the highlights
    keywords = [
        "Piton des Neiges",
        "Piton de la Fournaise",
        "Voile de la Mariée",
        "Plage de l’Hermitage, La Passe",
        "Plaine des Sables",
        "Trou de Fer",
        "Belvédère du Maïdo",
        "Gouffre de l'Etang-Salé",
        "Plage de Boucan Canot",
        "Maison Folio",
        "Takamaka",
        "Forêts de Bébour-Bélouve",
        "Route des Laves",
        "Anse des Cascades",
        "Vanilleraie (La)"
    ]
    df_best = df_poi[
        df_clean["Nom_du_POI"].apply(lambda x: any(k in x for k in keywords))
    ]
    df_best = df_best[-((df_best['Nom_du_POI'] == 'Salle Multimédia Piton des Neiges') | (
                df_best['Nom_du_POI'] == 'Fromagerie de Takamaka') | (
                                    df_best['Nom_du_POI'] == "Anse des Cascades (L')"))]

    # exclude the highlights from the df POI
    df_poi_3 = (
        pd.merge(df_poi_2, df_best, indicator=True, how="outer")
        .query('_merge=="left_only"')
        .drop("_merge", axis=1)
    )
    return df_clean, df_poi_3, df_food, df_outdooractivity, df_best, df_poi


def write_reg_reu_main_csv():
    # fix corrupted entries which are spread over several lines in the csv file by putting them on one line

    header_re = re.compile(r"Nom_du_POI")
    uri_re = re.compile(
        r"/[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}\n"
    )

    name, extension = os.path.splitext("datatourisme-reg-reu-main.csv")
    fixedFile = name + "_fixed" + extension

    with open("datatourisme-reg-reu-main.csv", encoding="utf-8") as f_in, open(
        fixedFile, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            if header_re.search(line) or uri_re.search(line):
                f_out.write(line)  # header and correct lines remain unchanged
            else:
                f_out.write(
                    line.rstrip() + ", "
                )  # incomplete lines: remove line return
    return fixedFile


@api.on_event("startup")
async def app_startup():
    ##################################
    #### extraction of categories ####
    ##################################

    # generation of hierarchical lists of categories according to https://www.datatourisme.fr/ontology/core/ontology.xml
    # the following output files are generated:
    # ontology_hierarchy_descending.csv (for sql)
    # allclasses.csv, subclass_class_relations.csv (for neo4j)

    file_content = Path("ontology.xml").read_text()
    root = ET.fromstring(file_content)

    df_cat_en_fr = compute_df_cat_en_fr(root)

    (
        master_categories,
        classes_to_be_removed_list,
        full_categories_chain_list,
        all_reversed_categories_chains_sorted,
    ) = get_classes_to_be_removed_list(root)

    write_files(master_categories, classes_to_be_removed_list)

    #################################
    ####### csv preprocessing #######
    #################################

    fixedFile = write_reg_reu_main_csv()

    (
        df_clean,
        df_poi_3,
        df_food,
        df_outdooractivity,
        df_best,
        df_poi,
    ) = prepare_dataframes(
        df_cat_en_fr,
        fixedFile,
        full_categories_chain_list,
        all_reversed_categories_chains_sorted,
    )
    api.state.df_clean = df_clean
    api.state.df_poi_3 = df_poi_3
    api.state.df_food = df_food
    api.state.df_outdooractivity = df_outdooractivity
    api.state.df_best = df_best
    api.state.df_poi = df_poi
    print("loaded")


@api.get("/itineraire", responses=responses)
async def get_itinerary(
    num_days: Optional[int] = Query(
        7,
        title="Number of days",
        description="Number of days. Accepted values: between 3 and 20",
    ),
    poi_per_day: Optional[int] = Query(
        4,
        title="Number of points of interest per day",
        description="Number of points of interest per day. Accepted values: between 2 and 10",
    ),
    food_per_day: Optional[int] = Query(
        2,
        title="Number of food places per day",
        description="Number of food places per day. Accepted values: between 0 and 10",
    ),
    activity_per_day: Optional[int] = Query(
        2,
        title="Number of actvities per day",
        description="Number of actvities per day. Accepted values: between 0 and 10",
    ),
):
    """
    Create an itinerary for your stay
    """

    if not 3 <= num_days <= 20:
        raise HTTPException(
            status_code=400, detail="Number of days should be between 3 and 20"
        )
    if not 2 <= poi_per_day <= 10:
        raise HTTPException(
            status_code=400,
            detail="Number of points of interest per day should be between 3 and 20",
        )
    if not 0 <= food_per_day <= 10:
        raise HTTPException(
            status_code=400,
            detail="Number of food places per day should be between 3 and 20",
        )
    if not 0 <= activity_per_day <= 10:
        raise HTTPException(
            status_code=400,
            detail="Number of activities per day should be between 3 and 20",
        )

    df_clean = api.state.df_clean
    df_poi_3 = api.state.df_poi_3
    df_food = api.state.df_food
    df_outdooractivity = api.state.df_outdooractivity
    df_best = api.state.df_best
    df_poi = api.state.df_poi

    #################################
    ########## clustering ###########
    #################################

    X = df_clean[["Longitude", "Latitude"]].values

    X_poi = df_poi_3[["Longitude", "Latitude"]].values

    X_food = df_food[["Longitude", "Latitude"]].values

    X_activity = df_outdooractivity[["Longitude", "Latitude"]].values

    X_best = df_best[["Longitude", "Latitude"]].values

    # Perform K-Means clustering to group the points of interest into clusters

    kmeans = KMeans(n_clusters=num_days)

    cluster_labels = kmeans.fit_predict(X)

    # Assign the principal point of interest and other points of interest for each cluster
    principal_pois = []
    other_pois = []
    activity = []
    food = []
    for i in range(num_days):
        cluster_pois = X[cluster_labels == i]
        cluster_principals = []
        cluster_food = []
        cluster_activity = []
        cluster_other = []
        for poi in cluster_pois:
            if poi in X_best:
                cluster_principals.append(poi)
            if poi in X_food:
                cluster_food.append(poi)
            if poi in X_activity:
                cluster_activity.append(poi)
            if poi in X_poi:
                cluster_other.append(poi)

        principal_pois.append(cluster_principals)
        activity.append(cluster_activity)
        food.append(cluster_food)
        other_pois.append(cluster_other)

    # Manually assign points of interest to each day based on proximity to principal point of interest
    result_principals = []
    result_other = []
    result_food = []
    result_activity = []

    for i in range(num_days):
        day_principals = principal_pois[i]
        day_food = []
        day_activity = []
        day_poi = []
        principals_centre = [sum(x) / len(x) for x in zip(*principal_pois)]

        if (
            principals_centre
        ):  # set the centre of principal poi as the 'center' of the cluster
            other_pois_sorted = sorted(
                other_pois[i], key=lambda x: np.linalg.norm(x - principals_centre)
            )
            food_sorted = sorted(
                food[i], key=lambda x: np.linalg.norm(x - principals_centre)
            )
            activity_sorted = sorted(
                activity[i], key=lambda x: np.linalg.norm(x - principals_centre)
            )
        else:
            activity_sorted = activity[i]
            food_sorted = food[i]
            other_pois_sorted = other_pois[i]

        day_poi.extend(other_pois_sorted[:poi_per_day])
        day_food.extend(food_sorted[:food_per_day])
        day_activity.extend(activity_sorted[:activity_per_day])

        result_principals.append(day_principals)
        result_other.append(day_poi)
        result_food.append(day_food)
        result_activity.append(day_activity)

    # Print the itinerary
    itinerary = []
    for j, principals in enumerate(result_principals):
        for p, principal in enumerate(principals):
            itinerary.append(
                df_best[
                    (df_best["Longitude"] == principal[0])
                    & (df_best["Latitude"] == principal[1])
                ].assign(Jour=j + 1, Type="Incontournable")
            )

    for k, others in enumerate(result_other):
        for o, other in enumerate(others):
            itinerary.append(
                df_poi[
                    (df_poi["Longitude"] == other[0]) & (df_poi["Latitude"] == other[1])
                ].assign(Jour=k + 1, Type="POI")
            )

    for l, foods in enumerate(result_food):
        for f, food in enumerate(foods):
            itinerary.append(
                df_food[
                    (df_food["Longitude"] == food[0]) & (df_food["Latitude"] == food[1])
                ].assign(Jour=l + 1, Type="Restaurant")
            )

    for m, activitys in enumerate(result_activity):
        for a, activity in enumerate(activitys):
            itinerary.append(
                df_outdooractivity[
                    (df_outdooractivity["Longitude"] == activity[0])
                    & (df_outdooractivity["Latitude"] == activity[1])
                ].assign(Jour=m + 1, Type="Activité")
            )

    df_itinerary = pd.concat(itinerary)
    df_itinerary = df_itinerary[
        [
            "Jour",
            "Type",
            "Nom_du_POI",
            "commune",
            "Adresse_postale",
            "Longitude",
            "Latitude",
            "superclass_0",
            "subclass_0",
            "superclass_1",
            "subclass_1",
        ]
    ].sort_values(by="Jour")

    df_itinerary = df_itinerary.fillna("")
    data = df_itinerary.to_dict(orient="records")

    return data
