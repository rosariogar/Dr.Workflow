#Carga de librerias
import os
import pandas as pd
import pandas_profiling as pf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report


def load_data(file_path):
    if file_path == '0':
        print("\nVolviendo al menú principal...")
        return None
    
    try:
        path = os.path.join('/app/data', file_path)
        df = pd.read_csv(path, sep=None, engine='python')
         
        # Solicitar al usuario el nombre del archivo de informe
        profile_name = input("\nIngrese como quiere llamar al informe inicial de los datos (sin extensión): ")
        # Generar un informe de perfil con el título personalizado
        profile = pf.ProfileReport(df, title=profile_name, explorative=True)
        profile_path = os.path.join('/app/resultados', f'{profile_name}.html')
        profile.to_file(profile_path)
        print(f"\nInforme de perfil generado en {profile_path}")
        print("\nArchivo cargado exitosamente")
        return df
    except Exception:
        print("\nError al cargar los datos")
        print("\nPor favor, verifica que has especificado bien la ruta al iniciar Docker")
        return None

def find_target_variable(df):
    # Verifica si el dataset es un DataFrame de pandas
    if not isinstance(df, pd.DataFrame):
        raise ValueError("\nEl dataset debe ser un DataFrame de pandas")
    
    unique_counts = df.nunique()

    # Encuentra las columnas que tienen el número mínimo de valores únicos (mayores a 0 y menores que 5)
    target_variables = unique_counts[(unique_counts > 0) & (unique_counts < 5)].index.tolist()

    return target_variables

def explore_target_variable(df, target_col):
    if target_col in df.columns:
        df_target = pd.DataFrame(df[target_col], columns=[target_col])
        value_counts = df_target[target_col].value_counts()
        n_values = len(value_counts)
        if len(value_counts) > 2:
            palette = sns.color_palette('pastel', n_colors=n_values)
            plt.bar(value_counts.index, value_counts.values, color=palette)
            plt.xlabel(target_col)
            plt.ylabel('Frecuencia')
            plt.title(f'Distribución de {target_col}')
            image_filename = input("¿Como quieres que se llame la imagen de la variable objetivo? ") + '.png'
            image_path = os.path.join('/app/resultados', image_filename)
            plt.savefig(image_path)
            plt.close()
            print(f"Imagen de la distribución de la clase {target_col} guardada en: {image_path}")
        else:
            sns.countplot(x=target_col, data=df, palette="bwr")
            plt.xlabel(target_col)
            plt.title(f'Distribución de {target_col}')
            image_filename = input("¿Como quieres que se llame la imagen de la variable objetivo? ") + '.png'
            image_path = os.path.join('/app/resultados', image_filename)
            plt.savefig(image_path)
            plt.close()
            print(f"Imagen de la distribución de la clase {target_col} guardada en: {image_path}")
    else:
        print(f"La columna {target_col} no se encuentra en el DataFrame.")

    if df[target_col].dtype == 'object':
        # Si la variable target es categórica
        # Obtener la frecuencia de cada categoría
        value_counts = df[target_col].value_counts()

        # Iterar sobre las categorías y calcular el porcentaje de pacientes en cada categoría
        for category in value_counts.index:
            count = len(df[df[target_col] == category])
            percent = count / len(df) * 100
            print(f"\nPORCENTAJE DE PACIENTES EN LA CATEGORÍA {category}: {percent:.2f}%")

    else:
        # Si la variable target es numérica
        # Obtener los valores únicos en la variable target
        unique_values = df[target_col].unique()

        # Iterar sobre los valores únicos y calcular el porcentaje de pacientes con cada valor
        for value in unique_values:
            count = len(df[df[target_col] == value])
            percent = count / len(df) * 100
            if percent > 0:
                print(f"\nPORCENTAJE DE PACIENTES CON {value}: {percent:.2f}%")


def explore_data(df):
    # Número de filas y columnas
    print("Tamaño del DataFrame:", df.shape)

    # Tipos de datos de cada columna
    print("\nTipo de datos:")
    print(df.dtypes)
    
    # Visualizar las primeras 5 filas
    print("\nPrimeras 5 filas del DataFrame:")
    print(df.head())

    # Información general del DataFrame
    print("\nInformación general del DataFrame:")
    print(df.info())

    # Estadísticas descriptivas del DataFrame
    print("\nEstadísticas descriptivas del DataFrame:")
    nonunique_columns = [col for col in df.columns if df[col].nunique() < len(df)]  #Identificar columnas no únicas
    df_nonunique = df[nonunique_columns]   #Filtrar DataFrame
    descriptive_stats = df_nonunique.describe() # Estadística descriptiva
    print(descriptive_stats)

    # Valores únicos de cada columna
    print("\nValores únicos de cada columna:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} valores únicos")

    # Valores faltantes por columna
    print("\nValores faltantes por columna:")
    print(df.isnull().sum())

    # Variable target
    target_variable = find_target_variable(df)
    target_variable_str = ', '.join(target_variable)
    print("\nLas posibles variables objetivo son: " + target_variable_str)


def clean_preprocess_data(df):
    # Eliminar filas duplicadas
    df.drop_duplicates(inplace=True)

    # Eliminar filas con valores faltantes a más del 25%
    missing_cols = []
    threshold = 0.25
    num_samples = len(df)
    for index, row in df.iterrows():
        missing_count = row.isnull().sum()
        missing_ratio = missing_count / len(row)
        if missing_ratio > threshold:
            missing_cols.append(index)

    df.drop(missing_cols, axis=0, inplace=True)

    # Elimina las columnas con valores faltantes a más del 50%
    missing_cols = []
    threshold = 0.5
    num_samples = len(df)
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_ratio = missing_count / num_samples
        if missing_ratio > threshold:
            missing_cols.append(col)

    df.drop(missing_cols, axis=1, inplace=True)

    # Eliminar las columnas con valores constantes
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_cols.append(col)

    df.drop(constant_cols, axis=1, inplace=True)

    # Convertir las columnas booleanas a 0 y 1
    boolean_cols = [col for col in df.columns if df[col].dtype == "bool"]
    df[boolean_cols] = df[boolean_cols].astype(np.int8)

    
    # Codificar las variables categóricas como variables numéricas
    categorical_cols = [col for col in df.columns if df[col].dtype == "object"]
    for col in categorical_cols:
        df[col] = df[col].str.upper() # convierte a MAYUSCULA
        if set(df[col].unique()) == {'YES', 'NO'}:
            df[col] = df[col].map({'YES': 1, 'NO': 0}) #POR SI ESTA EN INGLES
        elif set(df[col].unique()) == {'SI', 'NO'}:
            df[col] = df[col].map({'SI': 1, 'NO': 0}) #POR SI ESTA EN ESPAÑOL
        else:
            df[col] = pd.factorize(df[col])[0]
    
    # Reemplazar infinitos con NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # Calcular la media de cada columna
    column_means = df.mean()
    # Rellenar los valores NaN (anteriormente infinitos) con la media de cada columna
    df = df.fillna(column_means)


    # Solicitar al usuario el nombre del archivo de informe
    profile_name = input("\nIngrese como quiere llamar la tabla de la limpieza de datos (sin extensión): ")
    
    # Guardar el DataFrame limpio en un archivo HTML
    cleaned_html_path = os.path.join('/app/resultados', f'{profile_name}.html')
    df.to_html(cleaned_html_path, index=False)
    print(f"DataFrame limpio guardado en: {cleaned_html_path}")

    return df
            

def correlation_matrix(df):
    plt.figure(figsize=(20, 20))
    sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
    # Guardar la imagen en el sistema de archivos del contenedor
    image_filename = input("\nIngrese como quiere llamar a la imagen de la correlacion: ")  # Nombre del archivo
    image_path = os.path.join('/app/resultados', f'{image_filename}.png')
    plt.savefig(image_path)
    plt.close()
    print(f"Imagen guardada en: {image_path}")

    return image_path

def standardize_dataframe(df):
    continuous_cols = []
    unique_threshold = 20  # Criterio para considerar una variable como continua

    # Comprobar que variables son continuas
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            unique_values = df[col].unique()
            if len(unique_values) > unique_threshold and len(unique_values) < len(df):
                continuous_cols.append(col)
                
    #Estandarizamos las columnas continuas
    scaler = StandardScaler()
    df_selected = df[continuous_cols]
    df_selected_std = scaler.fit_transform(df_selected)
    df[continuous_cols] = df_selected_std
    
    # Solicitar al usuario el nombre del archivo de informe
    profile_name = input("\nIngrese como quiere llamar la tabla de la estandarizacion de datos (sin extensión): ")
    
    # Guardar el DataFrame limpio en un archivo HTML
    html_path = os.path.join('/app/resultados',f'{profile_name}.html')
    df.to_html(html_path, index=False)
    print(f"DataFrame estandarizado guardado en: {html_path}")

    return df   

def recommend_balancing(data, class_column, imbalance_threshold=0.2):
    # Contar las frecuencias de las clases
    class_counts = Counter(data[class_column])
    
    # Identificar la clase mayoritaria y minoritaria
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)
    
    # Calcular la proporción entre las clases
    majority_ratio = class_counts[majority_class] / len(data)
    minority_ratio = class_counts[minority_class] / len(data)
    
    # Verificar si hay un desequilibrio significativo
    if minority_ratio < imbalance_threshold * majority_ratio:
        print("El conjunto de datos está desbalanceado.")
        print(f"Proporción en clase mayoritaria ({majority_class}): {majority_ratio:.2f}")
        print(f"Proporción en clase minoritaria ({minority_class}): {minority_ratio:.2f}")
        print("Se recomienda considerar el balanceo de datos.")
    else:
         print("El conjunto de datos está bien balanceado.") 

def balance_dataset(data, class_column):
    # Contar las frecuencias de las clases
    class_counts = Counter(data[class_column])
    majority_class = max(class_counts, key=class_counts.get)
    
    # Separar las clases en diferentes DataFrames
    majority_class_data = data[data[class_column] == majority_class]
    minority_class_data = data[data[class_column] != majority_class]
    
    # Submuestrear la clase mayoritaria para que tenga la misma cantidad de ejemplos que la clase minoritaria
    majority_class_downsampled = resample(majority_class_data,
                                          replace=False,
                                          n_samples=len(minority_class_data),
                                          random_state=42)
    
    # Combinar los DataFrames balanceados
    balanced_data = pd.concat([majority_class_downsampled, minority_class_data])

   # Solicitar al usuario el nombre del archivo de informe
    profile_name = input("\nIngrese como quiere llamar la tabla de los datos balanceados (sin extensión): ")
    
    # Guardar el DataFrame limpio en un archivo HTML
    html_path = os.path.join('/app/resultados', f'{profile_name}.html')
    balanced_data.to_html(html_path, index=False)
    print(f"DataFrame balanceado guardado en: {html_path}")
    
    return balanced_data

def autoML(df, target_col, test_size=0.2, random_state=42):
    # Verificar si el nombre de la columna objetivo existe en el DataFrame
    if target_col not in df.columns:
        raise ValueError(f"La columna '{target_col}' no se encuentra en el DataFrame.")
        
    # Separar características y variable objetivo
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Verificar el número de clases únicas en la variable objetivo
    num_classes = len(y.unique())
    
    # Dividir el dataset en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Definir los modelos de clasificación adecuados según el número de clases
    if num_classes == 2:
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Support Vector Machine': SVC(probability=True)
        }
    else:
        models = {
            'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Support Vector Machine': SVC(probability=True)
        }
        
    # Entrenar y evaluar los modelos
    results = {}
    best_auc = 0
    best_model = None
    best_model_name = None

    for name, model in models.items():

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')  # Puedes ajustar el número de folds (cv)
        accuracy = scores.mean()  # Tomar el promedio de las puntuaciones
 
        # Entrenamiento
        model.fit(X_train, y_train)

        # Predicción
        y_pred = model.predict(X_test)

        # Evaluación
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

        # Métricas adicionales

        conf_matrix = confusion_matrix(y_test, y_pred)
        if num_classes == 2:
            tn, fp, fn, tp = conf_matrix.ravel()
            specificity = tn / (tn + fp) if (tn + fp).any() != 0 else 0
            fpr = fp / (fp + tn) if (fp + tn).any() != 0 else 0
        else:
            tn, fp, fn, tp = np.diag(conf_matrix), np.sum(conf_matrix, axis=0) - np.diag(conf_matrix), np.sum(conf_matrix, axis=1) - np.diag(conf_matrix), np.sum(conf_matrix) - (np.sum(conf_matrix, axis=0) + np.sum(conf_matrix, axis=1) - np.diag(conf_matrix))
            specificity = tn / (tn + fp) if (tn + fp).any() != 0 else np.zeros_like(tn)
            fpr = fp / (fp + tn) if (fp + tn).any() != 0 else np.zeros_like(fp)
        
        s_accuracy = (recall + specificity) / 2
        
         # Calcular AUC solo si es un problema de clasificación binaria
        if num_classes == 2:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr, tpr)

            #Actualizar el mejor modelo para la curva ROC
            if auc_score > best_auc:
                best_auc = auc_score
                best_model = model
                best_model_name = name
                fpr_best = fpr
                tpr_best = tpr

            results[name] = {
                'ACC': accuracy,
                'AUC': auc_score,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Specificity': specificity,
                'S-Accuracy': s_accuracy,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn
            }
        else:
            # Usar classification_report para obtener un resumen de métricas multiclase
            class_report = classification_report(y_test, y_pred, output_dict=True)
            f1 = class_report['weighted avg']['f1-score']
            results[name] = {
                'ACC': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn

            }

    # Guardar la curva ROC para el mejor modelo (solo si es clasificación binaria)
    if best_model is not None and num_classes == 2:
        # y_prob_best = best_model.predict_proba(X_test)[:, 1]
        auc_best = auc(fpr_best, tpr_best)
        plt.figure()
        plt.plot(fpr_best, tpr_best, color='darkorange', lw=2, label='Curva ROC (area = %0.2f)' % auc_best)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title(f'\nCurva ROC de {best_model_name}')
        image_name = input("\nIndica como quieres llamar a la imagen de la curva ROC del mejor modelo ")
        image_path = os.path.join('/app/resultados', f'{image_name}.png')
        plt.savefig(image_path)
        plt.close()
        

    # Crear una lista de nombres a partir de las claves del diccionario 'results'
    row_names = list(results.keys())
    # Crear un DataFrame con los valores del diccionario y los nombres de fila
    df_results = pd.DataFrame.from_dict(results, orient='index')
    # Agregar la columna de nombres de fila
    df_results.insert(0, 'Métricas', row_names)

    if num_classes == 2:
        print(f'\nImagen de Curva ROC guardada en: {image_path}')

    
    # Solicitar al usuario el nombre del archivo de informe
    profile_name = input("\nIngrese como quiere llamar el resumen de AutoML (sin extensión): ")
    
    # Guardar el DataFrame limpio en un archivo HTML
    html_path = os.path.join('/app/resultados', f'{profile_name}.html' )
    df_results.to_html(html_path, index=False)
    print(f"\nAutoML guardado en: {html_path}")
        
    return df_results