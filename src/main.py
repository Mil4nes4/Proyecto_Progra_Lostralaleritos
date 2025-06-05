from modelo import extraer_caras
from grabar import grabar_modelo
from entrenamiento import entrenar_modelo
from prueba import probar_modelo

def menu_options():
    print("1. Cargar modelo")
    print("2. Grabar video")
    print("3. Entrenar Modelo")
    print("4. Probar Modelo")
    print("5. Salir")
    return input("Seleccione una opción: ")

def menu():
    while True:
        opcion = menu_options()
        if opcion == "1":
            video_path = input("Ingrese la ruta del video: ")
            codigo = input("Ingrese el código del estudiante: ")
            extraer_caras(video_path, "faces", codigo)
        elif opcion == "2":
            grabar_modelo("faces", input("Ingrese el código del estudiante: "))
        elif opcion == "3":
            entrenar_modelo()
        elif opcion == "4":
            print("Presione q para salir de la ventana de video")
            probar_modelo()
        elif opcion == "5":
            print("Programa finalizado.")
            break
        else:
            print("Esta opción no existe, inténtalo de nuevo.")

if __name__ == '__main__':
    menu()