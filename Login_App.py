import flet as ft
import pymysql
import re
import datetime
import cv2
import os
import numpy as np
from lpips.pretrained_networks import resnet
from matplotlib import pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
from pydantic_core.core_schema import none_schema

# Cargar el modelo preentrenado
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='casia-webface').eval()


resultado_global = None
id_usuario = None
nombre_usuario = None
apellido_usuario = None

# declaracion de rutas, una para el directorio y otro para la foto en especifico

directorio_actual = os.path.dirname(os.path.abspath(__file__))
print(f"directorio actual : '{directorio_actual}'")
directorio = os.path.join(directorio_actual, 'foto_temporal')
print(f"directorio foto : '{directorio}'")
directorio_foto = os.path.join(directorio_actual, 'foto_temporal', 'foto.jpg')


myconection = pymysql.connect(host='localhost', user='root', passwd='', db='empleados')
cur = myconection.cursor()

def create_body(page: ft.Page) -> ft.Control:
    id_field = ft.TextField(
        width=280,
        height=40,
        hint_text='ID',
        border='underline',
        color='white',
        prefix_icon=ft.icons.LOCK,
        password=True,
    )

    def toggle_password(e):
        id_field.password = not id_field.password
        page.update()

    show_id_checkbox = ft.Checkbox(
        label='Mostrar ID',
        on_change=toggle_password
    )

    def iniciar_sesion(e):
        global id_usuario, nombre_usuario, apellido_usuario

        id = id_field.value
        cur.execute(f'select nombre, apellido from trabajadores where codigo = "{id}"')
        resultado_global = cur.fetchall()

        if resultado_global:
            id_usuario = id
            nombre_usuario = resultado_global[0][0]
            apellido_usuario = resultado_global[0][1]

            ahora = datetime.datetime.now()
            cur.execute(
                f'insert into registro_op (codigo, operacion, fecha) values ("{id_usuario}", "INGRESO DE USUARIO","{ahora}");')

            myconection.commit()

            page.snack_bar = ft.SnackBar(ft.Text("Bienvenido, su ID es correcto", color="green"))
            page.snack_bar.open = True
            page.update()
            page.go("/Menu")
        else:
            page.snack_bar = ft.SnackBar(ft.Text("Lo siento, su ID no se encuentra en la base de datos", color="red"))
            page.snack_bar.open = True
            page.update()

    # Función para cargar imágenes y extraer características
    def cargar_imagenes():
        cur.execute("SELECT codigo, foto FROM trabajadores")
        fotos = cur.fetchall()
        vectores = []


        for codigo, foto_blob in fotos:
            # Convertir el BLOB a una imagen
            nparr = np.frombuffer(foto_blob, np.uint8)
            imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Detectar el rostro
            aligned_faces = mtcnn(imagen)

            if aligned_faces is None:
                print(f"No se detectó ningún rostro en la imagen del trabajador con código {codigo}.")
                continue  # Cambiar a continue para seguir con el siguiente código

            print(f"Se detectó un rostro en la imagen del trabajador con código {codigo}.")

            if aligned_faces.dim() == 3:  # If only one face is detected
                aligned_faces = aligned_faces.unsqueeze(0)  # Add batch dimension

                # Obtener el embedding del rostro
            embedding = model(aligned_faces).detach()
            vectores.append((codigo, embedding))  # Almacenar el código y el embedding

        return vectores

    # Función para comparar una nueva imagen
    def comparar_imagenes(face_log, vectores_trabajadores):
        aligned_faces_log = mtcnn(face_log)


        if aligned_faces_log is None:
            print("No se detectó ningún rostro en la imagen capturada.")
            return None

        print("Se detectó un rostro en la imagen capturada.")

        if aligned_faces_log.dim() == 3:  # If only one face is detected
            aligned_faces_log= aligned_faces_log.unsqueeze(0)  # Add batch dimension

            # Obtener el embedding del rostro
        embedding_log = model(aligned_faces_log).detach()

        for codigo, embedding_reg in vectores_trabajadores:
            # Calcular la distancia euclidiana entre los embeddings
            distancia = (embedding_log - embedding_reg).norm().item()
            if distancia < 1.0:  # You can adjust the threshold for verification
                return codigo  # Retornar el código del trabajador
            else:
                print("Different persons")

        return None

    def Camara_Login(e):
        global id_usuario, nombre_usuario, apellido_usuario

        # Abriendo la cámara
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Camara")

        ruta_foto = os.path.join(directorio, "foto.jpg")

        # Condicional en caso de que no exista el directorio donde se guardará la foto

        if not os.path.exists(directorio):
            page.snack_bar = ft.SnackBar(ft.Text("Directorio no encontrado", color="red"))
            page.snack_bar.open = True
            page.update()
            return  # Salir de la función si el directorio no existe

        while True:
            # Capturar la foto
            ret, frame = cap.read()

            # Hacer una copia del frame para mostrar el mensaje
            frame_con_mensaje = frame.copy()

            # Ag regar el mensaje en la copia del frame
            cv2.putText(frame_con_mensaje, "ESPACIO para tomar la foto, ESCAPE para salir",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA)

            # Mostrar el frame con el mensaje en la ventana
            cv2.imshow("Camara", frame_con_mensaje)

            # Esperar a que el usuario presione la tecla ESPACIO para tomar la foto
            tecla = cv2.waitKey(1)
            if tecla == 32:  # Tecla ESPACIO
                # Guardar la imagen original sin el mensaje
                cv2.imwrite(ruta_foto, frame)

                page.snack_bar = ft.SnackBar(ft.Text("Foto guardada exitosamente", color="black"))
                page.snack_bar.open = True
                page.update()
                break
            # Salir si se presiona la tecla ESCAPE
            elif tecla == 27:
                break

        # Quitar la cámara
        cap.release()
        cv2.destroyAllWindows()

        # Cargar la imagen capturada
        face_log = cv2.imread(ruta_foto)

        # Cargar las imágenes de la base de datos
        vectores_trabajadores = cargar_imagenes()

        # Comparar la imagen capturada con las imágenes en la base de datos
        codigo_encontrado = comparar_imagenes(face_log, vectores_trabajadores)

        if codigo_encontrado:
            # Si se encuentra una coincidencia, obtener el nombre y apellido
            cur.execute(f'SELECT codigo, nombre, apellido FROM trabajadores WHERE codigo = "{codigo_encontrado}"')
            resultado_global = cur.fetchone()

            if resultado_global:
                id_usuario = resultado_global[0]
                nombre_usuario = resultado_global[1]
                apellido_usuario = resultado_global[2]

                ahora = datetime.datetime.now()
                cur.execute(
                    f'INSERT INTO registro_op (codigo, operacion, fecha) VALUES ("{id_usuario}", "INGRESO DE USUARIO FACE ID", "{ahora}");')
                myconection.commit()

                page.snack_bar = ft.SnackBar(ft.Text("¡Bienvenido!", color="green"))
                page.snack_bar.open = True
                page.update()
                page.go("/Menu")
        else:
            page.snack_bar = ft.SnackBar(ft.Text("¡Error! No se encontraron coincidencias.", color="red"))
            page.snack_bar.open = True
            page.update()

        # Borrar la imagen "foto.jpg" después de procesarla
        if os.path.exists(ruta_foto):
            os.remove(ruta_foto)

    return ft.Container(
        ft.Row([
            ft.Container(
                ft.Column(controls=[
                    ft.Container(
                        ft.Image(
                            src='150x150_2.png',
                            width=150,
                        ),
                        padding=ft.padding.only(110, 20)
                    ),
                    ft.Text(
                        'Iniciar Sesión',
                        width=360,
                        size=30,
                        weight='w900',
                        text_align='center'
                    ),
                    ft.Container(
                        id_field,
                        padding=ft.padding.only(20, 10)
                    ),
                    ft.Container(
                        show_id_checkbox,
                        padding=ft.padding.only(20, 10)
                    ),
                    ft.Container(
                        ft.ElevatedButton(
                            content=ft.Text(
                                'INICIAR',
                                color='white',
                                weight='w500',
                            ),
                            width=280,
                            bgcolor='black',
                            on_click=iniciar_sesion
                        ),
                        padding=ft.padding.only(25, 10)
                    ),
                    ft.Container(
                        ft.ElevatedButton(
                            content=ft.Text(
                                'Iniciar con Face ID',
                                color='white',
                                weight='w500',
                            ),
                            width=280,
                            bgcolor='orange',
                            on_click=Camara_Login
                        ),
                        padding=ft.padding.only(25, 10)
                    ),
                    ft.Container(
                        ft.Row([
                            ft.Text(
                                '¿No tiene una cuenta?'
                            ),
                            ft.TextButton(
                                'Crear una cuenta',
                                on_click=lambda _: page.go("/Registro_Usuario")
                            ),
                        ],
                            spacing=8),
                        padding=ft.padding.only(40)
                    )
                ],
                alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                ),
                bgcolor='#01112b',
                width=380,
                height=460,
                border_radius=20
            ),
        ],
        alignment=ft.MainAxisAlignment.SPACE_EVENLY,
        ),
        padding=10,
    )

def main(page: ft.Page):
    page.window_width = 800
    page.window_height = 520
    page.padding = 0
    page.vertical_alignment = "center"
    page.horizontal_alignment = "center"

    def route_change(e):
        if page.route == "/Menu":
            page.views.clear()
            page.views.append(
                ft.View(
                    "/Menu",
                    [
                        ft.Row(
                            [
                                ft.Container(
                                    ft.Column(controls=[
                                        ft.Container(
                                            ft.Image(
                                                src='150x150_2.png',
                                                width=150,
                                            ),
                                            padding=ft.padding.only(110, 0)
                                        ),
                                        ft.Text(
                                            f"Bienvenido/a {nombre_usuario} {apellido_usuario}",
                                            width=390,
                                            size=25,
                                            weight='w900',
                                            text_align='center'
                                        ),
                                        ft.Text(
                                            'Que desea realizar hoy?',
                                            width=390,
                                            size=15,
                                            weight='w900',
                                            text_align='center'
                                        ),
                                        ft.Container(
                                            ft.ElevatedButton(
                                                content=ft.Text(
                                                    'Ingresar Hora de Entrada',
                                                    color='white',
                                                    weight='w500',
                                                ),
                                                width=320,
                                                bgcolor='green',
                                                on_click=lambda _: page.go("/Hora_Entrada")
                                            ),
                                            padding=ft.padding.only(25, 10)
                                        ),
                                        ft.Container(
                                            ft.ElevatedButton(
                                                content=ft.Text(
                                                    'Ingresar Hora de Salida',
                                                    color='white',
                                                    weight='w500',
                                                ),
                                                width=320,
                                                bgcolor='orange',
                                                on_click=lambda _: page.go("/Hora_Salida")
                                            ),
                                            padding=ft.padding.only(25, 10)
                                        ),
                                        ft.Container(
                                            ft.ElevatedButton(
                                                content=ft.Text(
                                                    'Ver Registros',
                                                    color='white',
                                                    weight='w500',
                                                ),
                                                width=320,
                                                bgcolor='blue',
                                                on_click=lambda _: page.go("/Registro_Hora")
                                            ),
                                            padding=ft.padding.only(25, 10)
                                        ),
                                        ft.Container(
                                            ft.ElevatedButton(
                                                content=ft.Text(
                                                    'Salir',
                                                    color='white',
                                                    weight='w500',
                                                ),
                                                width=320,
                                                bgcolor='Red',
                                                on_click=lambda _: page.go("")
                                            ),
                                            padding=ft.padding.only(25, 10)
                                        ),
                                    ],
                                        alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                                    ),
                                    bgcolor='#01112b',
                                    width=380,
                                    height=460,
                                    border_radius=20
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ],
                )
            )
        elif page.route == "/Hora_Entrada":
            page.views.clear()
            horaEntrada = ""  # Variable para almacenar el contenido del TextField

            def guardar_hora(e):
                nonlocal horaEntrada  # Acceder a la variable hora_entrada desde la función guardar_hora
                horaEntrada = text_field.value  # Obtener el valor del TextField
                patron = r"^([01][0-9]|2[0-3]):([0-5][0-9])$"
                if re.match(patron, horaEntrada):
                    hora_dividida = horaEntrada.split(":")
                    hora = int(hora_dividida[0])
                    minutos = int(hora_dividida[1])
                    if hora < 0 or hora > 23 or minutos < 0 or minutos > 59:
                        page.snack_bar = ft.SnackBar(
                            ft.Text("La hora ingresada no es válida. Por favor, ingrese una hora entre 00:00 y 23:59",
                                    color="red"))
                    else:
                        cur.execute(
                            f'insert into asistencia (codigo, nombre, apellido, hora, tipo) values ("{id_usuario}", "{nombre_usuario }","{apellido_usuario}", "{horaEntrada}", "ENTRADA");')
                        myconection.commit()
                        page.snack_bar = ft.SnackBar(ft.Text(f"Hora registrada al: {horaEntrada}", color="green"))

                        ahora = datetime.datetime.now()
                        cur.execute(
                            f'insert into registro_op (codigo, operacion, fecha) values ("{id_usuario}", "REGISTRO HORA ENTRADA","{ahora}");')

                        myconection.commit()
                else:
                    page.snack_bar = ft.SnackBar(ft.Text(
                        "La hora ingresada no tiene el formato correcto. Por favor, ingrese una hora en formato HH:MM",
                        color="red"))
                page.snack_bar.open = True
                page.update()


            text_field = ft.TextField(
                width=280,
                height=40,
                hint_text='Hora de Entrada',
                border='underline',
                color='white',
            )

            page.views.append(
                ft.View(
                    "/Hora_Entrada",
                    [
                        ft.Row(
                            [
                                ft.Container(
                                    ft.Column(controls=[
                                        ft.Container(
                                            ft.Image(
                                                src='150x150_2.png',
                                                width=150,
                                            ),
                                            padding=ft.padding.only(110, 0)
                                        ),
                                        ft.Text(
                                            'Ingresar Hora de Entrada',
                                            width=390,
                                            size=30,
                                            weight='w900',
                                            text_align='center'
                                        ),
                                        ft.Text(
                                            'En hora militar, por ejemplo: 07:30',
                                            width=390,
                                            size=15,
                                            weight='w900',
                                            text_align='center'
                                        ),
                                        ft.Container(
                                            text_field,
                                            padding=ft.padding.only(20, 10)
                                        ),
                                        ft.Container(
                                            ft.ElevatedButton(
                                                content=ft.Text(
                                                    'Guardar',
                                                    color='white',
                                                    weight='w500',
                                                ),
                                                width=320,
                                                bgcolor='green',
                                                on_click=guardar_hora
                                            ),
                                            padding=ft.padding.only(25, 10)
                                        ),
                                        ft.Container(
                                            ft.ElevatedButton(
                                                content=ft.Text(
                                                    'Regresar',
                                                    color='white',
                                                    weight='w500',
                                                ),
                                                width=320,
                                                bgcolor='red',
                                                on_click=lambda _: page.go("/Menu")

                                            ),
                                            padding=ft.padding.only(25, 10)

                                        ),
                                    ],
                                        alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                                    ),
                                    bgcolor='#01112b',
                                    width=380,
                                    height=460,
                                    border_radius=20
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ],
                )
            )
        elif page.route == "/Hora_Salida":
            page.views.clear()
            horaSalida = ""  # Variable para almacenar el contenido del TextField

            def guardar_hora_salida(e):
                nonlocal horaSalida # Acceder a la variable hora_entrada desde la función guardar_hora
                horaSalida = text_field.value  # Obtener el valor del TextField
                patron = r"^([01][0-9]|2[0-3]):([0-5][0-9])$"
                if re.match(patron, horaSalida):
                    hora_dividida = horaSalida.split(":")
                    hora = int(hora_dividida[0])
                    minutos = int(hora_dividida[1])
                    if hora < 0 or hora > 23 or minutos < 0 or minutos > 59:
                        page.snack_bar = ft.SnackBar(
                            ft.Text("La hora ingresada no es válida. Por favor, ingrese una hora entre 00:00 y 23:59",
                                    color="red"))
                    else:
                        cur.execute(
                            f'insert into asistencia (codigo, nombre, apellido, hora, tipo) values ("{id_usuario}", "{nombre_usuario }","{apellido_usuario}", "{horaSalida}", "SALIDA");')
                        myconection.commit()
                        page.snack_bar = ft.SnackBar(ft.Text(f"Hora registrada al: {horaSalida}", color="green"))

                        ahora = datetime.datetime.now()
                        cur.execute(
                            f'insert into registro_op (codigo, operacion, fecha) values ("{id_usuario}", "REGISTRO HORA SALIDA","{ahora}");')

                        myconection.commit()
                else:
                    page.snack_bar = ft.SnackBar(ft.Text(
                        "La hora ingresada no tiene el formato correcto. Por favor, ingrese una hora en formato HH:MM",
                        color="red"))
                page.snack_bar.open = True
                page.update()


            text_field = ft.TextField(
                width=280,
                height=40,
                hint_text='Hora de Salida',
                border='underline',
                color='white',
            )

            page.views.append(
                ft.View(
                    "/Hora_Entrada",
                    [
                        ft.Row(
                            [
                                ft.Container(
                                    ft.Column(controls=[
                                        ft.Container(
                                            ft.Image(
                                                src='150x150_2.png',
                                                width=150,
                                            ),
                                            padding=ft.padding.only(110, 0)
                                        ),
                                        ft.Text(
                                            'Ingresar Hora de Salida',
                                            width=390,
                                            size=30,
                                            weight='w900',
                                            text_align='center'
                                        ),
                                        ft.Text(
                                            'En hora militar, por ejemplo: 17:30',
                                            width=390,
                                            size=15,
                                            weight='w900',
                                            text_align='center'
                                        ),
                                        ft.Container(
                                            text_field,
                                            padding=ft.padding.only(20, 10)
                                        ),
                                        ft.Container(
                                            ft.ElevatedButton(
                                                content=ft.Text(
                                                    'Guardar',
                                                    color='white',
                                                    weight='w500',
                                                ),
                                                width=320,
                                                bgcolor='green',
                                                on_click=guardar_hora_salida
                                            ),
                                            padding=ft.padding.only(25, 10)
                                        ),
                                        ft.Container(
                                            ft.ElevatedButton(
                                                content=ft.Text(
                                                    'Regresar',
                                                    color='white',
                                                    weight='w500',
                                                ),
                                                width=320,
                                                bgcolor='red',
                                                on_click=lambda _: page.go("/Menu")
                                            ),
                                            padding=ft.padding.only(25, 10)
                                        ),
                                    ],
                                        alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                                    ),
                                    bgcolor='#01112b',
                                    width=380,
                                    height=460,
                                    border_radius=20
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ],
                )
            )
        elif page.route == "/Registro_Hora":
            page.views.clear()

            registros_list = ft.ListView(
                controls=[],
                divider_thickness=1,
                height=200,
                expand=True,
            )

            page.views.append(
                ft.View(
                    "/Hora_Entrada",
                    [
                        ft.Row(
                            [
                                ft.Container(
                                    ft.Column(controls=[
                                        ft.Container(
                                            ft.Image(
                                                src='150x150_2.png',
                                                width=150,
                                            ),
                                            padding=ft.padding.only(110, 0)
                                        ),
                                        ft.Text(
                                            'Registros de horarios',
                                            width=390,
                                            size=30,
                                            weight='w900',
                                            text_align='center'
                                        ),
                                        registros_list,
                                        ft.Container(
                                            ft.ElevatedButton(
                                                content=ft.Text(
                                                    'Regresar',
                                                    color='white',
                                                    weight='w500',
                                                ),
                                                width=320,
                                                bgcolor='red',
                                                on_click=lambda _: page.go("/Menu")
                                            ),
                                            padding=ft.padding.only(25, 10)
                                        ),
                                    ],
                                        alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                                    ),
                                    bgcolor='#01112b',
                                    width=380,
                                    height=460,
                                    border_radius=20
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ],
                )
            )

            cur.execute(f'select nombre, apellido, hora, tipo from asistencia where codigo = "{id_usuario}"')
            result = cur.fetchall()

            ahora = datetime.datetime.now()
            cur.execute(
                f'insert into registro_op (codigo, operacion, fecha) values ("{id_usuario}", "MUESTRA ASISTENCIAS","{ahora}");')

            for nombre, apellido, hora, tipo in result:
                registros_list.controls.append(
                    ft.ListTile(
                        title=ft.Text(f"{nombre} {apellido} - {hora} - {tipo}"),
                        leading=ft.Icon(ft.icons.PERSON),
                    )
                )

            page.update()

        elif page.route == "/Registro_Usuario":

            import cv2
            import os

            def camara_Registro(e):
                # Abriendo la cámara
                cap = cv2.VideoCapture(0)
                cv2.namedWindow("Camara")

                # Condicional en caso de que no exista el directorio donde se guardará la foto
                if not os.path.exists(directorio):
                    page.snack_bar = ft.SnackBar(ft.Text("Directorio no encontrado", color="red"))
                    page.snack_bar.open = True
                    page.update()
                    return  # Salir de la función si el directorio no existe

                while True:
                    # Capturar la foto
                    ret, frame = cap.read()

                    # Hacer una copia del frame para mostrar el mensaje
                    frame_con_mensaje = frame.copy()

                    # Agregar el mensaje en la copia del frame
                    cv2.putText(frame_con_mensaje, "ESPACIO para tomar la foto, ESCAPE para salir",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 0),
                                2,
                                cv2.LINE_AA)

                    # Mostrar el frame con el mensaje en la ventana
                    cv2.imshow("Camara", frame_con_mensaje)

                    # Esperar a que el usuario presione la tecla ESPACIO para tomar la foto
                    tecla = cv2.waitKey(1)
                    if tecla == 32:  # Tecla ESPACIO
                        # Guardar la imagen original sin el mensaje
                        ruta_foto = os.path.join(directorio, "foto.jpg")
                        cv2.imwrite(ruta_foto, frame)

                        # Detectar rostros en la imagen
                        faces = mtcnn(frame)

                        # Verificar si se detectaron rostros
                        if faces is None or len(faces) == 0:
                            print("No se detectaron rostros. Eliminando la foto.")
                            os.remove(ruta_foto)  # Eliminar la foto si no hay rostros
                            page.snack_bar = ft.SnackBar(
                                ft.Text("No se detectaron rostros. Foto eliminada.", color="red"))
                            page.snack_bar.open = True
                            page.update()
                        else:
                            # Si se detectan rostros, puedes continuar con el resto del programa
                            page.snack_bar = ft.SnackBar(ft.Text("Foto guardada exitosamente", color="black"))
                            page.snack_bar.open = True
                            page.update()
                            break  # Salir del bucle si se guardó la foto

                    # Salir si se presiona la tecla ESCAPE
                    elif tecla == 27:
                        break

                # Quitar la cámara
                cap.release()
                cv2.destroyAllWindows()

            def salir_y_borrar(e):
                page.go("")
                if os.path.exists(directorio):
                    # itera sobre todos los archivos en el directorio
                    for archivo in os.listdir(directorio):
                        ruta_archivo = os.path.join(directorio, archivo)
                        try:
                            if os.path.isfile(ruta_archivo):  # Verificar si es un archivo
                                os.remove(ruta_archivo)  # Borrar el archivo
                            elif os.path.isdir(ruta_archivo):  # Verificar si es un directorio
                                os.rmdir(ruta_archivo)  # Borrar el directorio (vacío)
                        except Exception as e:
                            print(f"mensaje en caso de no encontrar nada que borrar, Razón: {e}")

            page.views.clear()
            nombre = ""
            apellido = ""
            cedula_usuario = ""

            def guardar_usuario(e):
                nonlocal nombre, apellido, cedula_usuario
                nombre = text_field_nombre.value
                apellido = text_field_apellido.value
                cedula_usuario = text_field_codigo.value

                # Verificar si todos los campos están llenos
                if not nombre or not apellido or not cedula_usuario:
                    page.snack_bar = ft.SnackBar(ft.Text("Por favor, complete todos los campos", color="red"))
                    page.snack_bar.open = True
                    page.update()
                    return

                # Verificar que la cédula tenga al menos 3 dígitos
                if len(cedula_usuario) < 3:
                    page.snack_bar = ft.SnackBar(ft.Text("La cédula debe tener al menos 3 dígitos", color="red"))
                    page.snack_bar.open = True
                    page.update()
                    return
                # Verifica si la carpeta donde se guarda la foto esta vacia
                if not os.listdir(directorio):
                    page.snack_bar = ft.SnackBar(ft.Text("Para completar el registro debe tomarse la foto", color="red"))
                    page.snack_bar.open = True
                    page.update()
                    return

                # Verificar que la cédula sea un número válido
                try:
                    cedula_usuario = int(cedula_usuario)
                except ValueError:
                    page.snack_bar = ft.SnackBar(ft.Text("Invalido. Ingrese un número válido", color="red"))
                    page.snack_bar.open = True
                    page.update()
                    return

                with open(directorio_foto, 'rb') as file:
                    foto = file.read()

                ultimos_digitos = str(cedula_usuario)[-3:]
                codigo = f"{nombre[0]}{apellido[0]}{ultimos_digitos}"

                valores= codigo, nombre, apellido, cedula_usuario, foto
                sql= '''
        INSERT INTO trabajadores (codigo, nombre, apellido, cedula, foto) 
        VALUES (%s, %s, %s, %s, %s)
    '''
                cur.execute(sql,valores)

                ahora = datetime.datetime.now()
                cur.execute(
                    f'insert into registro_op (codigo, operacion, fecha) values ("{codigo}", "CREACION DE USUARIO","{ahora}");')

                myconection.commit()

                page.snack_bar = ft.SnackBar(
                    ft.Text(f"Usuario registrado, su codigo es: {codigo}, inicie sesion para ingresar", color="black"))
                page.snack_bar.open = True
                page.update()
                page.go("")

            text_field_nombre = ft.TextField(
                width=280,
                height=40,
                hint_text='Nombre',
                border='underline',
                color='white',
            )
            text_field_apellido = ft.TextField(
                width=280,
                height=40,
                hint_text='Apellido',
                border='underline',
                color='white',
            )
            text_field_codigo = ft.TextField(
                width=280,
                height=40,
                hint_text='Cedula',
                border='underline',
                color='white',
            )
            page.views.append(
                ft.View(
                    "/Registros_Usuarios",
                    [
                        ft.Row(
                            [
                                ft.Container(
                                    ft.Column(controls=[
                                        ft.Container(
                                            ft.Image(
                                                src='150x150_2.png',
                                                width=150,
                                            ),
                                            padding=ft.padding.only(110, 0)
                                        ),
                                        ft.Text(
                                            'Registro de Usuario',
                                            width=390,
                                            size=25,
                                            weight='w900',
                                            text_align='center'
                                        ),
                                        ft.Container(
                                            text_field_nombre,
                                            padding=ft.padding.only(20, 10)
                                        ),
                                        ft.Container(
                                            text_field_apellido,
                                            padding=ft.padding.only(20, 10)
                                        ),
                                        ft.Container(
                                            text_field_codigo,
                                            padding=ft.padding.only(20, 10)
                                        ),
                                        ft.Container(
                                            ft.ElevatedButton(
                                                content=ft.Text(
                                                    'Capturar Rostro',
                                                    color='white',
                                                    weight='w500',
                                                ),
                                                width=320,
                                                bgcolor='orange',
                                                on_click=camara_Registro
                                            ),
                                            padding=ft.padding.only(25, 10)
                                        ),
                                        ft.Container(
                                            ft.ElevatedButton(
                                                content=ft.Text(
                                                    'Guardar',
                                                    color='white',
                                                    weight='w500',
                                                ),
                                                width=320,
                                                bgcolor='green',
                                                on_click=guardar_usuario
                                            ),
                                            padding=ft.padding.only(25, 10)
                                        ),
                                        ft.Container(
                                            ft.ElevatedButton(
                                                content=ft.Text(
                                                    'Regresar',
                                                    color='white',
                                                    weight='w500',
                                                ),
                                                width=320,
                                                bgcolor='red',
                                                on_click=lambda _: salir_y_borrar(page)
                                            ),
                                            padding=ft.padding.only(25, 10)
                                        ),
                                    ],
                                        alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                                    ),
                                    bgcolor='#01112b',
                                    width=380,
                                    height=460,
                                    border_radius=20
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ],
                )
            )

        elif page.route == "":
            page.views.clear()
            page.views.append(
                ft.View(
                    "",
                    [create_body(page)]
                )
            )
        page.update()

    page.on_route_change = route_change
    page.add(create_body(page))

ft.app(target=main, view = WEB_BROWSER)