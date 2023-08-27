import os
from componentes import *
import pandas as pd
def main():
    print("\nBienvenido!, soy Dr. Workflow. Te ayudaré a realizar un análisis de los datos de tus pacientes")
    # Configurar la visualización para mostrar todas las filas y columnas
    pd.set_option('display.max_rows', None)  # Mostrar todas las filas
    pd.set_option('display.max_columns', None)  # Mostrar todas las columna

    while True:
        print("\n Por favor, seleccione una opción:")
        print("1. Cargar datos y creación de informe detallado sobre el dataframe")
        print("2. Mostrar por consola un resumen de la exploración inicial y las posibles variable objetivo")
        print("3. Realizar análisis de variable objetivo")
        print("4. Limpiar los datos")
        print("5. Guardar la matriz de correlación en PNG")
        print("6. Estandarizar los datos" )
        print("7. Comprobar si los datos estan balanceados" )
        print("8. Balancear dataset")
        print("9. Realizar análisis automático (AutoML)")
        print("10. Salir")

        choice = input("Ingrese el número de la opción: ")

        if choice == "1":
            file_path = input("Ingrese el nombre del archivo que desea analizar: ")
            df = load_data(file_path)
            clean_df = None
            target_col = None
            standardized_df  = None


        elif choice == "2":
            if 'df' not in locals():
                print("\nPrimero debe cargar los datos utilizando la opción 1.")
                continue
            explore_data(df)
        elif choice == "3":
            if 'df' not in locals():
                    print("\nPrimero debe cargar los datos utilizando la opción 1.")
                    continue
            target_col = input("\nIngrese el nombre de la columna objetivo: ")
            if target_col not in df.columns:
                    print(f"La columna '{target_col}' no existe en el DataFrame.")
            else:
                explore_target_variable(df, target_col)
                
        elif choice == "4": 
            if 'df' not in locals():
                print("\nPrimero debe cargar los datos utilizando la opcion 1")
                continue
            clean_df = clean_preprocess_data(df)
        
        elif choice == "5": 
             if 'df' not in locals():
                print("\nPrimero debe cargar los datos utilizando la opción 1.")
                continue
             if 'df' in locals() and clean_df is None:
                print("\nPrimero debe limpiar y preprocesar los datos utilizando la opción 4")
                continue
             correlation_matrix(clean_df)
        
        elif choice == "6": 
             if 'df' not in locals():
                print("\nPrimero debe cargar los datos utilizando la opción 1.")
                continue
             if 'df' in locals() and clean_df is None:
                 print("\nPrimero debe limpiar y preprocesar los datos utilizando la opción 4")
                 continue
             standardized_df  = standardize_dataframe(clean_df)
             print("Datos estandarizados correctamente")

        elif choice == "7": 
             if 'df' not in locals():
                print("\nPrimero debe cargar los datos utilizando la opción 1.")
                continue
             
             if 'df' in locals() and clean_df is None:
                 print("\nLe recomiendo que primero que limpie y preprocese los datos ")
                 if target_col is None: 
                     target_col = input("Ingrese el nombre de la columna que contiene las clases: ")
                 recommend_balancing(df, target_col, imbalance_threshold=0.2)
                 continue
             
             if 'df' in locals() and clean_df is not None and standardized_df is None:
                 print("\nLe recomiendo que estandarice los datos, pero este seria el resultado: ")
                 if target_col is None: 
                     target_col = input("Ingrese el nombre de la columna que contiene las clases: ")
                 recommend_balancing(df, target_col, imbalance_threshold=0.2)
                 continue
             if 'df' in locals() and standardized_df  is not None:
                if target_col is None: 
                    target_col = input("\nIngrese el nombre de la columna que contiene las clases: ")
                recommend_balancing(df, target_col, imbalance_threshold=0.2)
                continue
                    
        elif choice == "8":
            if 'df' not in locals():
                print("\nPrimero debe cargar los datos utilizando la opción 1.")
                continue
             
            if 'df' in locals() and clean_df is None:
                 print("\nLe recomiendo que primero que limpie y preprocese los datos ")
                 if target_col is None: 
                     target_col = input("\nIngrese el nombre de la columna que contiene las clases: ")
                 balance = balance_dataset(df, target_col)
                 print("\nDataset balanceado con exito")
                 continue
             
            if 'df' in locals() and clean_df is not None and standardized_df  is None:
                 print("\nLe recomiendo que estandarice los datos ")
                 if target_col is None: 
                     target_col = input("\nIngrese el nombre de la columna que contiene las clases: ")
                 balance_dataset(clean_df, target_col)
                 print(balance)
                 print("\nDataset balanceado con exito")
                 continue
            if 'df' in locals() and standardized_df  is not None:
                if target_col is None: 
                    target_col = input("\nIngrese el nombre de la columna que contiene las clases: ")
                balance = balance_dataset(standardized_df , target_col)
                print("\nDataset balanceado con exito")
                continue

        elif choice == "9":
            if 'df' not in locals():
                print("\nPrimero debe cargar los datos utilizando la opción 1.")
                continue

            if 'df' in locals() and clean_df is None:
                 print("\nPrimero, al menos, debe limpiar los datos")
                 continue
             
            if 'df' in locals() and clean_df is not None and standardized_df  is None:
                 print("\nLe recomiendo que estandarice los datos ")
                 if target_col is None: 
                     target_col = input("Ingrese el nombre de la columna que contiene las clases: ")
                 result = autoML(clean_df, target_col)
                 print(result)
                 print("\nAnálisis automático completado")
                 continue
            if 'df' in locals() and standardized_df  is not None:
                if target_col is None: 
                    target_col = input("\nIngrese el nombre de la columna que contiene las clases: ")
                result = autoML(standardized_df, target_col)
                print("\nAnálisis automático completado")
                continue

        elif choice == "10":
            print("\nSaliendo del programa...")
            break
        else:
            print("\nOpción inválida. Por favor, elija una opción válida.")

if __name__ == "__main__":
    main()