import cv2
import numpy as np
import math

def calcular_distancia(x1, y1, x2, y2):
    """Calcula la distancia entre dos puntos."""
    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distancia

def dibujar_distancias(imagen, puntos, escala):
    """Calcula y dibuja las distancias entre los puntos en la imagen."""

    punto_anterior = None
    for punto in puntos:
        if punto_anterior is not None:
            # Calcular la distancia entre los puntos
            distancia = calcular_distancia(punto_anterior[0], punto_anterior[1], punto[0], punto[1]) * escala

            # Calcular la posición media para dibujar el texto
            x_medio = (punto_anterior[0] + punto[0]) // 2
            y_medio = (punto_anterior[1] + punto[1]) // 2

            # Dibujar la distancia en la imagen
            cv2.putText(imagen, f"{distancia:.1f}m", (x_medio, y_medio), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        punto_anterior = punto

def detectar_playa_y_dibujar_contorno(imagen_entrada, imagen_salida, num_divisiones, archivo_salida, escala):
    # Abrir el archivo de salida
    with open(archivo_salida, 'w') as archivo:
        archivo.write("Mapa1\n")

    # Cargar la imagen
    imagen = cv2.imread(imagen_entrada)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Convertir la imagen a espacio de color HSV (más robusto para variaciones de color)
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir una gama extendida de colores para la arena en HSV
    lower_color1 = np.array([20, 20, 80])   # Tonos más oscuros de la arena
    upper_color1 = np.array([30, 120, 230]) 
    lower_color2 = np.array([0, 20, 90])    # Tonos más claros (amarillentos)
    upper_color2 = np.array([35, 80, 250])

    # Crear máscaras para detectar píxeles en ambas gamas de arena
    mascara1 = cv2.inRange(imagen_hsv, lower_color1, upper_color1)
    mascara2 = cv2.inRange(imagen_hsv, lower_color2, upper_color2)

    # Combinar las máscaras
    mascara = cv2.bitwise_or(mascara1, mascara2)

    # Suavizar la máscara para eliminar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mascara_limpia = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_OPEN, kernel)

    # Eliminar regiones verdes (vegetación), rojas y azules
    lower_verde1 = np.array([30, 20, 20])  # Gama extendida del verde claro
    upper_verde1 = np.array([50, 255, 255])
    lower_verde2 = np.array([50, 20, 20])  # Verde medio
    upper_verde2 = np.array([70, 255, 255])
    lower_verde3 = np.array([70, 20, 20])  # Verde más oscuro
    upper_verde3 = np.array([95, 255, 255])
    lower_verde4 = np.array([0, 255, 52])  # Verde más oscuro
    upper_verde4 = np.array([92,110, 68])
    lower_rojo1 = np.array([0, 70, 50])
    upper_rojo1 = np.array([10, 255, 255])
    lower_rojo2 = np.array([170, 70, 50])
    upper_rojo2 = np.array([180, 255, 255])
    lower_azul = np.array([90, 50, 50])
    upper_azul = np.array([130, 255, 255])

    # Crear máscaras para colores no deseados
    mascara_verde1 = cv2.inRange(imagen_hsv, lower_verde1, upper_verde1)
    mascara_verde2 = cv2.inRange(imagen_hsv, lower_verde2, upper_verde2)
    mascara_verde3 = cv2.inRange(imagen_hsv, lower_verde3, upper_verde3)
    mascara_verde4 = cv2.inRange(imagen_hsv, lower_verde4, upper_verde4)
    mascara_verde = cv2.bitwise_or(mascara_verde1, cv2.bitwise_or(mascara_verde2,  cv2.bitwise_or(mascara_verde3, mascara_verde4)))
    mascara_rojo = cv2.bitwise_or(cv2.inRange(imagen_hsv, lower_rojo1, upper_rojo1), 
                                  cv2.inRange(imagen_hsv, lower_rojo2, upper_rojo2))
    mascara_azul = cv2.inRange(imagen_hsv, lower_azul, upper_azul)
    mascara_colores_no_deseados = cv2.bitwise_or(mascara_verde, cv2.bitwise_or(mascara_rojo, mascara_azul))

    # Excluir los colores no deseados
    mascara_limpia = cv2.bitwise_and(mascara_limpia, cv2.bitwise_not(mascara_colores_no_deseados))

    # Crear una máscara para excluir bordes externos
    altura, ancho = mascara_limpia.shape
    margen_borde = 10  # Define un margen alrededor de los bordes
    mascara_sin_bordes = np.zeros_like(mascara_limpia)
    mascara_sin_bordes[margen_borde:altura-margen_borde, margen_borde:ancho-margen_borde] = 255

    # Aplicar la máscara de exclusión de bordes
    mascara_limpia = cv2.bitwise_and(mascara_limpia, mascara_sin_bordes)

    # Nueva máscara para excluir objetos pequeños
    mascara_objetos_grandes = np.zeros_like(mascara_limpia)
    contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 5000:  # Filtrar áreas pequeñas
            cv2.drawContours(mascara_objetos_grandes, [contorno], -1, 255, -1)

    # Actualizar la máscara limpia
    mascara_limpia = cv2.bitwise_and(mascara_limpia, mascara_objetos_grandes)

    # Encontrar contornos en la máscara final
    contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno más grande
    contorno_mas_grande = max(contornos, key=cv2.contourArea, default=None)

    # Dibujar solo el contorno más grande si existe
    if contorno_mas_grande is not None:
        area = cv2.contourArea(contorno_mas_grande)
        if area > 5000:  # Filtrar áreas pequeñas que no corresponden a la playa
            cv2.drawContours(imagen, [contorno_mas_grande], -1, (0, 0, 255), 7)  # Dibujar en rojo

            # Guardar las coordenadas del contorno rojo
            with open(archivo_salida, 'a') as archivo:
                archivo.write("--Coordenadas contorno rojo\n")
                for punto in contorno_mas_grande:
                    archivo.write(f"{punto[0][0]}, {punto[0][1]}\n")

            # Obtener los límites del contorno
            x, y, w, h = cv2.boundingRect(contorno_mas_grande)

            # Dividir el área en partes iguales por X coordenadas
            division_ancho = w // num_divisiones

            # Guardar la coordenada y del punto anterior para calcular la distancia
            punto_y_anterior = None
            punto_x_anterior = None
            centro_y_anterior = None

            # Guardar las coordenadas de los puntos para calcular distancias
            puntos_verdes = []
            puntos_interseccion = []

            for i in range(1, num_divisiones):
                punto_x = x + i * division_ancho
                for punto_y in range(y, y + h):
                    if cv2.pointPolygonTest(contorno_mas_grande, (punto_x, punto_y), False) >= 0:
                        cv2.line(imagen, (punto_x, y), (punto_x, y + h), (255, 0, 0), 2)
                        with open(archivo_salida, 'a') as archivo:
                            archivo.write(f"Zona{i}\n")
                            archivo.write(f"--Coordenadas intersección rojo/azul\n")
                            archivo.write(f"{punto_x}, {punto_y}\n")

                        if punto_y_anterior is not None:
                            # Calcular la distancia entre los puntos de intersección rojo/azul
                            distancia_puntos_interseccion = calcular_distancia(punto_x_anterior, punto_y_anterior, punto_x, punto_y) * escala
                            cv2.putText(imagen, f"{distancia_puntos_interseccion:.2f}m", (punto_x_anterior + (punto_x - punto_x_anterior) // 2, punto_y_anterior + (punto_y - punto_y_anterior) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            with open(archivo_salida, 'a') as archivo:
                                archivo.write(f"--Distancia entre puntos de intersección rojo/azul (metros): {distancia_puntos_interseccion:.2f}\n")

                        punto_y_anterior = punto_y
                        punto_x_anterior = punto_x
                        
                        # Guardar punto de intersección
                        puntos_interseccion.append((punto_x, punto_y))

                        break

                # Calcular la posición del punto verde en el centro de cada división
                centro_x = x + (i - 0.5) * division_ancho
                for centro_y in range(y, y + h):
                    if cv2.pointPolygonTest(contorno_mas_grande, (centro_x, centro_y), False) >= 0:
                        cv2.circle(imagen, (int(centro_x), centro_y + 20), 10, (0, 255, 0), -1)  # Desplazar el punto 20 píxeles hacia abajo
                        with open(archivo_salida, 'a') as archivo:
                            archivo.write(f"--Coordenada punto verde\n")
                            archivo.write(f"{int(centro_x)}, {centro_y + 20}\n")

                        if centro_y_anterior is not None:
                            # Calcular la distancia entre los puntos verdes
                            distancia_puntos_verdes = calcular_distancia(int(centro_x), centro_y + 20, int(centro_x), centro_y_anterior + 20) * escala
                            cv2.putText(imagen, f"{distancia_puntos_verdes:.2f}m", (int(centro_x) + 10, (centro_y + 20 + centro_y_anterior + 20) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            with open(archivo_salida, 'a') as archivo:
                                archivo.write(f"--Distancia entre puntos verdes (metros): {distancia_puntos_verdes:.2f}\n")

                        centro_y_anterior = centro_y
                        
                        # Guardar punto verde
                        puntos_verdes.append((int(centro_x), centro_y + 20))
                        
                        break

            # Añadir el punto adicional al final
            final_x = x + w - division_ancho // 2
            for final_y in range(y, y + h):
                if cv2.pointPolygonTest(contorno_mas_grande, (final_x, final_y), False) >= 0:
                    cv2.circle(imagen, (int(final_x), final_y + 20), 10, (0, 255, 0), -1)  # Desplazar el punto 20 píxeles hacia abajo

                    # Calcular la distancia entre los puntos verdes
                    distancia_puntos_verdes = calcular_distancia(int(final_x), final_y + 20, int(centro_x), centro_y_anterior + 20) * escala
                    cv2.putText(imagen, f"{distancia_puntos_verdes:.2f}m", (int(final_x) + 10, (final_y + 20 + centro_y_anterior + 20) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    with open(archivo_salida, 'a') as archivo:
                        archivo.write(f"--Coordenada punto verde\n")
                        archivo.write(f"{int(final_x)}, {final_y + 20}\n")
                        archivo.write(f"--Distancia entre puntos verdes (metros): {distancia_puntos_verdes:.2f}\n")
                        
                    # Guardar el último punto verde
                    puntos_verdes.append((int(final_x), final_y + 20))
                    
                    break
                    
            # Calcular y dibujar distancias entre puntos verdes
            dibujar_distancias(imagen, puntos_verdes, escala)

            # Calcular y dibujar distancias entre puntos de intersección
            dibujar_distancias(imagen, puntos_interseccion, escala)

    # Guardar la imagen de salida
    cv2.imwrite(imagen_salida, imagen)

# Solicitar al usuario el número de divisiones
num_divisiones = int(input("Ingrese el número de divisiones: "))

# Pedir al usuario que ingrese la escala
escala = float(input("Ingrese la escala (metros por píxel): "))

# Archivos de entrada y salida
imagen_entrada = "playa0.png"  # Archivo de entrada cargado
imagen_salida = "playa0_contorno_dividido.png"  # Archivo de salida

# Ejecutar la funcion
detectar_playa_y_dibujar_contorno(imagen_entrada, imagen_salida, num_divisiones, "zonas0.txt", escala)
print(f"Análisis completado. La imagen con el contorno dividido se ha guardado como {imagen_salida}.")

# Archivos de entrada y salida
imagen_entrada = "playa1.png"  # Archivo de entrada cargado
imagen_salida = "playa1_contorno_dividido.png"  # Archivo de salida

# Ejecutar la funcion
detectar_playa_y_dibujar_contorno(imagen_entrada, imagen_salida, num_divisiones, "zonas1.txt", escala)
print(f"Análisis completado. La imagen con el contorno dividido se ha guardado como {imagen_salida}.")

# Archivos de entrada y salida
imagen_entrada = "playa2.png"  # Archivo de entrada cargado
imagen_salida = "playa2_contorno_dividido.png"  # Archivo de salida

# Ejecutar la funcion
detectar_playa_y_dibujar_contorno(imagen_entrada, imagen_salida, num_divisiones, "zonas2.txt", escala)
print(f"Análisis completado. La imagen con el contorno dividido se ha guardado como {imagen_salida}.")



def analizar_imagen(imagen_entrada, imagen_salida):
    # Cargar la imagen
    imagen = cv2.imread(imagen_entrada)
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Dimensiones de la imagen
    altura, ancho, _ = imagen.shape
    total_pixeles = altura * ancho

    # Máscara para detectar agua (azul)
    lower_azul = np.array([90, 50, 50])
    upper_azul = np.array([130, 255, 255])
    mascara_agua = cv2.inRange(imagen_hsv, lower_azul, upper_azul)

    # Máscara para detectar arena utilizando los rangos originales
    lower_color1 = np.array([20, 20, 80])   # Tonos más oscuros de la arena
    upper_color1 = np.array([30, 120, 230])
    lower_color2 = np.array([0, 20, 90])    # Tonos más claros (amarillentos)
    upper_color2 = np.array([35, 80, 250])

    # Crear máscaras para detectar píxeles en ambas gamas de arena
    mascara_arena1 = cv2.inRange(imagen_hsv, lower_color1, upper_color1)
    mascara_arena2 = cv2.inRange(imagen_hsv, lower_color2, upper_color2)
    mascara_arena = cv2.bitwise_or(mascara_arena1, mascara_arena2)

    # Máscara para detectar zonas urbanas (resto de los colores no clasificados)
    mascara_colores_no_deseados = cv2.bitwise_or(mascara_agua, mascara_arena)
    mascara_urbana = cv2.bitwise_not(mascara_colores_no_deseados)

    # Calcular el porcentaje de cada categoría
    pixeles_agua = cv2.countNonZero(mascara_agua)
    pixeles_arena = cv2.countNonZero(mascara_arena)
    pixeles_urbana = cv2.countNonZero(mascara_urbana)

    porcentaje_agua = (pixeles_agua / total_pixeles) * 100
    porcentaje_arena = (pixeles_arena / total_pixeles) * 100
    porcentaje_urbana = (pixeles_urbana / total_pixeles) * 100

    # Imprimir los resultados
    print(f"Porcentaje de agua: {porcentaje_agua:.2f}%")
    print(f"Porcentaje de arena: {porcentaje_arena:.2f}%")
    print(f"Porcentaje de zona urbana: {porcentaje_urbana:.2f}%")

    # Pintar las zonas en la imagen original
    resultado = imagen.copy()

    # Crear una máscara para excluir el color blanco puro
    mascara_blanco = cv2.inRange(imagen, np.array([0, 0, 255]), np.array([0, 0, 255]))
    mascara_no_blanco = cv2.bitwise_not(mascara_blanco)

    # Pintar agua (azul)
    azul_overlay = np.zeros_like(imagen, dtype=np.uint8)
    azul_overlay[mascara_agua > 0] = (255, 0, 0)
    azul_overlay = cv2.bitwise_and(azul_overlay, azul_overlay, mask=mascara_no_blanco)
    resultado = cv2.addWeighted(resultado, 0.7, azul_overlay, 0.3, 0)

    # Pintar arena (rojo)
    rojo_overlay = np.zeros_like(imagen, dtype=np.uint8)
    rojo_overlay[mascara_arena > 0] = (0, 0, 180)
    rojo_overlay = cv2.bitwise_and(rojo_overlay, rojo_overlay, mask=mascara_no_blanco)
    resultado = cv2.addWeighted(resultado, 0.7, rojo_overlay, 0.3, 0)

    # Pintar zonas urbanas (verde)
    verde_overlay = np.zeros_like(imagen, dtype=np.uint8)
    verde_overlay[mascara_urbana > 0] = (0, 255, 0)
    verde_overlay = cv2.bitwise_and(verde_overlay, verde_overlay, mask=mascara_no_blanco)
    resultado = cv2.addWeighted(resultado, 0.7, verde_overlay, 0.3, 0)

    # Escribir los porcentajes en la imagen
    texto_agua = f"Aigua: {porcentaje_agua:.2f}%"
    texto_arena = f"Sorra: {porcentaje_arena:.2f}%"
    texto_urbana = f"Urbana: {porcentaje_urbana:.2f}%"

    posicion_x = ancho - 300
    posicion_y = altura - 30
    cv2.putText(resultado, texto_agua, (posicion_x, posicion_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(resultado, texto_arena, (posicion_x, posicion_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(resultado, texto_urbana, (posicion_x, posicion_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Guardar la imagen resultante
    cv2.imwrite(imagen_salida, resultado)

# Archivo de entrada y salida
imagen_entrada = "playa0_contorno_dividido.png"
imagen_salida = "playa0_analisis.png"

# Ejecutar la función
analizar_imagen(imagen_entrada, imagen_salida)

# Archivo de entrada y salida
imagen_entrada = "playa1_contorno_dividido.png"
imagen_salida = "playa1_analisis.png"

# Ejecutar la función
analizar_imagen(imagen_entrada, imagen_salida)

# Archivo de entrada y salida
imagen_entrada = "playa2_contorno_dividido.png"
imagen_salida = "playa2_analisis.png"

# Ejecutar la función
analizar_imagen(imagen_entrada, imagen_salida)


